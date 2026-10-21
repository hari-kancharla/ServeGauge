(function () {
  "use strict";
  const picker = document.getElementById("model-filter");
  const rows = Array.from(document.querySelectorAll(".endpoint-row"));
  const count = document.getElementById("visible-count");
  function filter() {
    if (!picker) return;
    let visible = 0;
    for (const row of rows) {
      row.hidden = picker.value !== "all" && row.dataset.model !== picker.value;
      if (!row.hidden) visible += 1;
    }
    if (count) count.textContent = "Showing " + visible + " of " + rows.length + " recorded endpoint captures";
  }
  if (picker) picker.addEventListener("change", filter);
  filter();
  const appearance = document.getElementById("appearance");
  if (appearance) {
    appearance.addEventListener("change", function () {
      document.documentElement.dataset.theme = appearance.value;
    });
  }
})();
