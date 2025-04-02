const express = require('express');
const multer = require('multer');
const cors = require('cors');
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');

const app = express();
const port = process.env.PORT || 10000;

// CORS 설정
app.use(cors());
app.use(express.json());

// 임시 파일 저장을 위한 multer 설정
const storage = multer.diskStorage({
  destination: function(req, file, cb) {
    cb(null, 'uploads/');
  },
  filename: function(req, file, cb) {
    cb(null, Date.now() + path.extname(file.originalname));
  }
});

// 업로드 폴더가 없으면 생성
if (!fs.existsSync('uploads')) {
  fs.mkdirSync('uploads');
}

const upload = multer({ storage: storage });

// 로컬 모델 로드
let model;
async function loadModel() {
  try {
    // 로컬 파일 시스템에서 모델 로드
    const modelPath = 'file://' + __dirname + '/model/model.json';
    model = await tf.loadLayersModel(modelPath);
    console.log('모델이 성공적으로 로드되었습니다.');
    
    // 모델 클래스 정보 로드 (metadata.json이 있는 경우)
    try {
      const metadata = require('./model/metadata.json');
      if (metadata && metadata.labels) {
        console.log('모델 클래스:', metadata.labels);
      }
    } catch (metadataError) {
      console.log('메타데이터 파일이 없거나 로드할 수 없습니다. 기본 클래스명을 사용합니다.');
    }
  } catch (error) {
    console.error('모델 로드 중 오류 발생:', error);
  }
}

// 서버 시작 시 모델 로드
loadModel();

// 이미지 전처리 함수
async function preprocessImage(imagePath) {
  // 이미지를 읽어서 텐서로 변환
  const imageBuffer = fs.readFileSync(imagePath);
  const tfImage = tf.node.decodeImage(imageBuffer);
  
  // Teachable Machine 모델은 일반적으로 224x224 크기의 이미지를 사용합니다
  const resized = tf.image.resizeBilinear(tfImage, [224, 224]);
  
  // 정규화 (0-1 범위로)
  const normalized = resized.div(255.0);
  
  // 배치 차원 추가
  const batched = normalized.expandDims(0);
  
  return batched;
}

// 이미지 업로드 및 추론 엔드포인트
app.post('/predict', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: '이미지가 없습니다.' });
    }

    if (!model) {
      return res.status(500).json({ error: '모델이 로드되지 않았습니다.' });
    }

    // 이미지 전처리
    const imageTensor = await preprocessImage(req.file.path);
    
    // 추론 실행
    const predictions = await model.predict(imageTensor);
    const results = await predictions.data();
    
    // 추론 결과를 클래스 이름과 확률로 변환
    // metadata.json에서 클래스 이름 로드 (없으면 기본값 사용)
    let classNames;
    try {
      const metadata = require('./model/metadata.json');
      classNames = metadata.labels || ['class1', 'class2', 'class3'];
    } catch (e) {
      // 메타데이터 파일이 없으면 기본 클래스명 사용
      classNames = ['class1', 'class2', 'class3'];
    }
    
    const resultArray = Array.from(results).map((probability, index) => {
      return {
        class: classNames[index] || `class${index + 1}`,
        probability: probability
      };
    });
    
    // 확률이 높은 순으로 정렬
    resultArray.sort((a, b) => b.probability - a.probability);
    
    // 임시 파일 삭제
    fs.unlinkSync(req.file.path);
    
    // JSON 결과 반환
    res.json({
      success: true,
      predictions: resultArray,
      topPrediction: resultArray[0]
    });
    
  } catch (error) {
    console.error('추론 중 오류 발생:', error);
    res.status(500).json({ error: '추론 중 오류가 발생했습니다.', details: error.message });
  }
});

// 서버 상태 확인용 엔드포인트
app.get('/', (req, res) => {
  res.json({ status: 'online', message: 'Teachable Machine 추론 서버가 실행 중입니다.' });
});

app.listen(port, () => {
  console.log(`서버가 포트 ${port}에서 실행 중입니다.`);
});
