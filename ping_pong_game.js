let playerPaddle, aiPaddle, ball;
let playerScore = 0, aiScore = 0;
let ballSpeedX, ballSpeedY;
let PADDLE_WIDTH, PADDLE_HEIGHT, BALL_WIDTH, BALL_HEIGHT, PADDLE_SPEED, BALL_SPEED_X, BALL_SPEED_Y;

function setup() {
  createCanvas(windowWidth, windowHeight);

  PADDLE_WIDTH = 0.03 * windowWidth;
  PADDLE_HEIGHT = 0.2 * windowHeight;
  BALL_WIDTH = 0.03 * windowWidth;
  BALL_HEIGHT = 0.03 * windowWidth;
  PADDLE_SPEED = 0.02 * windowHeight;
  BALL_SPEED_X = 0.008 * windowWidth;
  BALL_SPEED_Y = 0.008 * windowWidth;

  playerPaddle = createVector(0.05 * windowWidth, height / 2 - PADDLE_HEIGHT / 2);
  aiPaddle = createVector(width - 0.05 * windowWidth - PADDLE_WIDTH, height / 2 - PADDLE_HEIGHT / 2);
  ball = createVector(width / 2 - BALL_WIDTH / 2, height / 2 - BALL_HEIGHT / 2);

  ballSpeedX = BALL_SPEED_X * (random() < 0.5 ? 1 : -1);
  ballSpeedY = BALL_SPEED_Y * (random() < 0.5 ? 1 : -1);
}

function draw() {
  background(255);

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
  textSize(0.05 * windowWidth);
  text(`Player: ${playerScore}`, 0.05 * windowWidth, 0.07 * windowHeight);
  text(`AI: ${aiScore}`, width - 0.15 * windowWidth, 0.07 * windowHeight);
}

// Resize canvas when the window size changes
function windowResized() {
  resizeCanvas(windowWidth, windowHeight);

  // Adjust paddle and ball positions and sizes when the window size changes
  playerPaddle.x = 0.05 * windowWidth;
  playerPaddle.y = height / 2 - PADDLE_HEIGHT / 2;
  aiPaddle.x = width - 0.05 * windowWidth - PADDLE_WIDTH;
  aiPaddle.y = height / 2 - PADDLE_HEIGHT / 2;
  ball.x = width / 2 - BALL_WIDTH / 2;
  ball.y = height / 2 - BALL_HEIGHT / 2;
}
