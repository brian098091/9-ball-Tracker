let fileName = ""
document.getElementById('videoUploader').addEventListener('change', function(event) {
    const file = event.target.files[0];
    fileName = file['name']
    const url = URL.createObjectURL(file);
    const videoPlayer = document.getElementById('videoPlayer');
    videoPlayer.src = url;
    videoPlayer.play();
  
    const videoContainer = document.querySelector('.video-container');
    videoContainer.style.display = 'flex';
  });
  
  document.getElementById('getFrameBtn').addEventListener('click', function() {
    const videoPlayer = document.getElementById('videoPlayer');
    const canvas = document.getElementById('frameCanvas');
    const context = canvas.getContext('2d');
    
    canvas.width = videoPlayer.videoWidth;
    canvas.height = videoPlayer.videoHeight;
    context.drawImage(videoPlayer, 0, 0, canvas.width, canvas.height);

    videoPlayer.pause();
  });
  
  document.getElementById('frameCanvas').addEventListener('click', function(event) {
    const canvas = document.getElementById('frameCanvas');
    const context = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const pixel = context.getImageData(x, y, 1, 1).data;
    const rgb = `rgb(${pixel[0]}, ${pixel[1]}, ${pixel[2]})`;

    document.getElementById('coordinates').textContent = `X: ${x.toFixed(2)}, Y: ${y.toFixed(2)}`;


    document.getElementById('rValue').textContent = `R: ${pixel[0]}`;
    document.getElementById('gValue').textContent = `G: ${pixel[1]}`;
    document.getElementById('bValue').textContent = `B: ${pixel[2]}`;

    document.getElementById('selectedColor').style.backgroundColor = rgb;

    document.getElementById('colorInfo').style.display = 'block';
});
  
document.getElementById('confirmColorBtn').addEventListener('click', function() {
  const currentTime = document.getElementById('videoPlayer').currentTime;
  const VideoMSEC = Math.floor(currentTime * 1000); 

  const coordinatesText = document.getElementById('coordinates').textContent;
  const coordinates = coordinatesText.match(/X: (\d+.\d+), Y: (\d+.\d+)/);
  const xCoordinate = parseFloat(coordinates[1]);
  const yCoordinate = parseFloat(coordinates[2]);

  const H = document.getElementById('gap1').value;
  const S = document.getElementById('gap2').value;
  const V = document.getElementById('gap3').value;

  fetch('http://127.0.0.1:5000/process_frame', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      msec: VideoMSEC,
      x: xCoordinate,
      y: yCoordinate,
      h:H,
      s:S,
      v:V,
      file_name:fileName
    })
  })
  .then(response => response.json())
  .then(data => {
    const maskImage = document.getElementById('maskImage');
    const maskContainer = document.getElementById('maskContainer');

    maskImage.src = data.mask_image; 
    maskContainer.style.display = 'block'; 
    console.log(data);
  })
  .catch(error => {
    console.error('Error:', error);
  });
});
  