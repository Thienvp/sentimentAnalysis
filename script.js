document.getElementById('demoForm').addEventListener('submit', function (event) {
    event.preventDefault();
  
    // Lấy giá trị từ trường văn bản
    var inputText = document.getElementById('textInput').value;
  
    // Gửi yêu cầu HTTP POST đến API /predict
    axios.post('http://127.0.0.1/predict', {
        text: inputText
      })
      .then(function (response) {
        // Hiển thị kết quả từ API
        var resultDiv = document.getElementById('result');
        var sentimentText = document.getElementById('sentimentText');
        sentimentText.innerText = 'Message: ' + response.data.message + ', Score: ' + response.data.score;
        resultDiv.classList.remove('hidden');
      })
      .catch(function (error) {
        // Xử lý lỗi nếu có
        console.error('Lỗi:', error);
      });
  });
  
