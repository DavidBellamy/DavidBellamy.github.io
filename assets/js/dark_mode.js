document.addEventListener("DOMContentLoaded", function () {
  const toggle = document.getElementById("light-toggle");
  if (!toggle) return;

  toggle.addEventListener("click", () => {
    const current = localStorage.getItem("theme") || "light";
    setTheme(current === "light" ? "dark" : "light");
  });
});
