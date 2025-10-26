document.addEventListener("DOMContentLoaded", () => {
  const setupDiv = document.getElementById("quiz-setup");
  const quizContainer = document.getElementById("quiz-content");
  const backBtn = document.getElementById("back-btn");
  const startBtn = document.getElementById("start-quiz");

  const text = localStorage.getItem("quizText");

  if (!text) {
    quizContainer.innerHTML = "<p class='error'>No text found for quiz generation.</p>";
    setupDiv.style.display = "none";
    quizContainer.style.display = "block";
    return;
  }

  startBtn.addEventListener("click", async () => {
    const count = parseInt(document.getElementById("question-count").value);
    const type = "MCQ"; // fixed for now

    setupDiv.style.display = "none";
    quizContainer.style.display = "block";

    // Fetch quiz questions from backend
    const questions = await getQuizQuestions(text, count, type);

    if (questions.length === 0) {
      quizContainer.innerHTML = "<p class='error'>Failed to generate quiz questions.</p>";
      return;
    }

    renderQuiz(questions, text, count, type);
  });

  backBtn.addEventListener("click", () => {
    window.close();
  });
});

/**
 * Async function to call Flask endpoint and get LLM-generated MCQs
 */
async function getQuizQuestions(text, count, type = "MCQ") {
  try {
    const quizContainer = document.getElementById("quiz-content");
    const tempMessage = document.createElement("div");
    tempMessage.classList.add("quiz-message");
    tempMessage.textContent = "Generating quiz questions...";
    quizContainer.appendChild(tempMessage);
    quizContainer.scrollTop = quizContainer.scrollHeight;

    const response = await fetch("https://backend-ktnv.onrender.com/generate_quiz", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, count, type }),
    });

    const data = await response.json();
    console.log("Prompt is:",{text});

    tempMessage.remove();

    return data.questions || [];
  } catch (err) {
    console.error("Error fetching quiz questions:", err);
    return [];
  }
}

/**
 * Render MCQs in the DOM and handle scoring
 */
function renderQuiz(questions, text, count, type) {
  const quizContainer = document.getElementById("quiz-content");

  quizContainer.innerHTML = `
    <div class="selected-text">
      <h2>Selected Text:</h2>
      <p>${text}</p>
    </div>
    <h2>Quiz (${count} MCQs)</h2>
    <form id="quiz-form">
      ${questions
        .map(
          (q, i) => `
        <div class="quiz-question">
          <p>${i + 1}. ${q.question_text}</p>
          ${q.options
            .map(
              (opt, j) => `
            <label>
              <input type="radio" name="q${i}" value="${opt}">
              ${opt}
            </label>
          `
            )
            .join("")}
        </div>
      `
        )
        .join("")}
      <button type="submit" id="submit-btn">Submit Answers</button>
    </form>
    <div id="result"></div>
  `;

  document.getElementById("quiz-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const result = document.getElementById("result");
    let score = 0;

    questions.forEach((q, i) => {
      const options = document.querySelectorAll(`input[name="q${i}"]`);
      const selected = document.querySelector(`input[name="q${i}"]:checked`);

      options.forEach((opt) => {
        const label = opt.parentElement;

        // Highlight correct answer
        if (opt.value === q.correct_answer) {
          label.style.backgroundColor = "lightgreen";
        } 
        // Highlight wrong selection
        if (selected && selected.value === opt.value && opt.value !== q.correct_answer) {
          label.style.backgroundColor = "salmon";
        }

        // Disable all options after submit
        opt.disabled = true;
      });

      if (selected && selected.value === q.correct_answer) score++;
    });

    result.innerHTML = `<p>You got ${score} out of ${questions.length} correct!</p>`;
  });
}

