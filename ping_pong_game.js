let playerPaddle, aiPaddle, ball;
let playerScore = 0, aiScore = 0;
let ballSpeedX, ballSpeedY;
const PADDLE_WIDTH = 15, PADDLE_HEIGHT = 100;
const BALL_WIDTH = 20, BALL_HEIGHT = 20;
const PADDLE_SPEED = 7, BALL_SPEED_X = 1, BALL_SPEED_Y = 1;
let gameStarted = false;

function setup() {
  createCanvas(800, 600);

  playerPaddle = createVector(30, height / 2 - PADDLE_HEIGHT / 2);
  aiPaddle = createVector(width - 30 - PADDLE_WIDTH, height / 2 - PADDLE_HEIGHT / 2);
  ball = createVector(width / 2 - BALL_WIDTH / 2, height / 2 - BALL_HEIGHT / 2);

  ballSpeedX = BALL_SPEED_X * (random() < 0.5 ? 1 : -1);
  ballSpeedY = BALL_SPEED_Y * (random() < 0.5 ? 1 : -1);
}

function draw() {
  background(255);

  if (gameStarted) {
    if (!isPaused) {
      // Draw paddles and ball
      fill(0);
      rect(playerPaddle.x, playerPaddle.y, PADDLE_WIDTH, PADDLE_HEIGHT);
      rect(aiPaddle.x, aiPaddle.y, PADDLE_WIDTH, PADDLE_HEIGHT);
      ellipse(ball.x + BALL_WIDTH / 2, ball.y + BALL_HEIGHT / 2, BALL_WIDTH, BALL_HEIGHT);

      // Move paddles and ball
      if (keyIsDown(87) && playerPaddle.y > 0) {
        playerPaddle.y -= PADDLE_SPEED;
      }
      if (keyIsDown(83) && playerPaddle.y < height - PADDLE_HEIGHT) {
        playerPaddle.y += PADDLE_SPEED;
      }

      if (ball.y < 0 || ball.y > height - BALL_HEIGHT) {
        ballSpeedY *= -1;
      }

      // AI movement
      if (ball.y + BALL_HEIGHT / 2 < aiPaddle.y + PADDLE_HEIGHT / 2) {
        aiPaddle.y -= PADDLE_SPEED;
      }
      if (ball.y + BALL_HEIGHT / 2 > aiPaddle.y + PADDLE_HEIGHT / 2) {
        aiPaddle.y += PADDLE_SPEED;
      }

      ball.x += ballSpeedX;
      ball.y += ballSpeedY;

      // Ball collision with paddles
      if (ball.x < playerPaddle.x + PADDLE_WIDTH && ball.x + BALL_WIDTH > playerPaddle.x &&
        ball.y < playerPaddle.y + PADDLE_HEIGHT && ball.y + BALL_HEIGHT > playerPaddle.y) {
        ballSpeedX *= -1;
      }

      if (ball.x < aiPaddle.x + PADDLE_WIDTH && ball.x + BALL_WIDTH > aiPaddle.x &&
        ball.y < aiPaddle.y + PADDLE_HEIGHT && ball.y + BALL_HEIGHT > aiPaddle.y) {
        ballSpeedX *= -1;
      }

      // Ball goes out of bounds
      if (ball.x < 0 || ball.x > width) {
        if (ball.x < 0) {
          aiScore++;
        } else {
          playerScore++;
        }

        ballSpeedX = BALL_SPEED_X * (random() < 0.5 ? 1 : -1);
        ballSpeedY = BALL_SPEED_Y * (random() < 0.5 ? 1 : -1);
        ball = createVector(width / 2 - BALL_WIDTH / 2, height / 2 - BALL_HEIGHT / 2);
      }

      // Display scores
      fill(0);
      textSize(40);
      text(`Player: ${playerScore}`, 20, 40);
      text(`AI: ${aiScore}`, width - 120, 40);
    } else {
      // Game is paused, display countdown timer
      fill(0);
      textSize(24);
      textAlign(CENTER);
      let countdown = Math.ceil((pauseEndTime - millis()) / 1000);
      text(`Game Paused - Resume in ${countdown} seconds`, width / 2, height / 2);
    }
  } 
}

function startGame() {
  gameStarted = true;
  let startButton = select('#startButton');
  startButton.hide();
}

function pauseGame() {
  if (!isPaused) {
    isPaused = true;
    let pauseButton = select('#pauseButton');
    pauseButton.html('Resume Game');
    pauseEndTime = millis() + pauseDuration;
    noLoop();
  } else {
    isPaused = false;
    let pauseButton = select('#pauseButton');
    pauseButton.html('Pause Game');
    loop();
  }
}
