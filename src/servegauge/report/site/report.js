"use strict";
(() => {
  const payloadNode = document.getElementById("report-data");
  if (!payloadNode) return;
  const payload = JSON.parse(payloadNode.textContent);
  const rows = Array.from(document.querySelectorAll(".check-row"));
  const search = document.getElementById("check-search");
  const endpoint = document.getElementById("endpoint-filter");
  const outcome = document.getElementById("outcome-filter");
  const form = document.querySelector(".filter-bar");
  const chart = document.getElementById("outcomes-chart");
  const tableBody = document.getElementById("chart-data-body");
  const order = payload.legend.map(row => row.verdict);
  const labels = new Map(payload.endpoints.map(row => [row.id, row.label]));
  let view = null;
  let chartQueue = Promise.resolve();

  function countsFor(visibleRows) {
    const counts = new Map();
    for (const row of visibleRows) {
      const key = JSON.stringify([row.dataset.endpoint, row.dataset.verdict]);
      const value = counts.get(key) || {endpoint:row.dataset.endpoint, label:labels.get(row.dataset.endpoint) || row.dataset.endpoint, verdict:row.dataset.verdict, count:0, order:order.indexOf(row.dataset.verdict)};
      value.count += 1;
      counts.set(key, value);
    }
    return Array.from(counts.values()).sort((a,b) => a.label.localeCompare(b.label) || a.order-b.order);
  }

  function showChartTable(counts) {
    const fragment = document.createDocumentFragment();
    for (const row of counts) {
      const tr = document.createElement("tr");
      const label = document.createElement("th");
      label.scope = "row";
      label.textContent = row.label;
      const verdict = document.createElement("td");
      verdict.textContent = row.verdict;
      const count = document.createElement("td");
      count.className = "numeric";
      count.textContent = String(row.count);
      tr.append(label, verdict, count);
      fragment.append(tr);
    }
    tableBody.replaceChildren(fragment);
  }

  async function showChart(counts) {
    if (!chart || !window.vega) {
      if (chart) chart.dataset.chartState = "unavailable";
      return;
    }
    const width = Math.max(130, chart.clientWidth-142);
    const height = Math.max(90, new Set(counts.map(row => row.endpoint)).size*34);
    if (!counts.length) {
      chart.hidden = true;
      return;
    }
    chart.hidden = false;
    if (!view) {
      const spec = {
        width, height, padding:{left:119,right:12,top:10,bottom:40}, autosize:"none",
        background:"transparent",
        data:[{name:"counts",values:counts,transform:[{type:"stack",groupby:["label"],sort:{field:"order"},field:"count",as:["x0","x1"]}]}],
        scales:[
          {name:"x",type:"linear",domain:{data:"counts",field:"x1"},range:"width",nice:true,zero:true},
          {name:"y",type:"band",domain:{data:"counts",field:"label"},range:"height",padding:.32},
          {name:"color",type:"ordinal",domain:order,range:payload.legend.map(row => row.color)}
        ],
        axes:[
          {orient:"bottom",scale:"x",title:"Recorded checks (count)",grid:true,gridColor:"#edf1e9",domain:false,tickColor:"#d4dfd5",labelColor:"#5d706b",titleColor:"#5d706b",labelFontSize:10,titleFontSize:10,titlePadding:10,tickCount:5},
          {orient:"left",scale:"y",domain:false,ticks:false,labelColor:"#3d584b",labelFontSize:10,labelPadding:10,labelLimit:111}
        ],
        marks:[{type:"rect",from:{data:"counts"},encode:{update:{x:{scale:"x",field:"x0"},x2:{scale:"x",field:"x1"},y:{scale:"y",field:"label"},height:{scale:"y",band:1},fill:{scale:"color",field:"verdict"},cornerRadius:{value:2},description:{signal:"datum.label + ': ' + datum.verdict + ', ' + datum.count + ' recorded checks'"}}}}]
      };
      const loader = vega.loader();
      loader.load = async () => {throw new Error("Report charts use embedded data only");};
      loader.sanitize = async () => {throw new Error("Report charts do not load external resources");};
      view = new vega.View(vega.parse(spec), {renderer:"svg",container:chart,hover:false,loader});
      await view.runAsync();
    } else {
      view.data("counts",counts).width(width).height(height);
      await view.runAsync();
    }
    chart.dataset.chartState = "ready";
  }

  function applyFilters() {
    const query = search.value.trim().toLowerCase();
    const visible = [];
    for (const row of rows) {
      const shown = (endpoint.value === "all" || row.dataset.endpoint === endpoint.value)
        && (outcome.value === "all" || row.dataset.verdict === outcome.value ||
          (outcome.value === "other" && ["INFO","NOT_SUPPORTED","NOT_APPLICABLE"].includes(row.dataset.verdict)))
        && (!query || row.dataset.search.toLowerCase().includes(query));
      row.hidden = !shown;
      if (shown) visible.push(row);
    }
    document.getElementById("visible-count").textContent = `Showing ${visible.length} of ${rows.length} check records`;
    document.getElementById("no-results").hidden = visible.length !== 0;
    const counts = countsFor(visible);
    showChartTable(counts);
    chartQueue = chartQueue.then(() => showChart(counts)).catch(() => {
      chart.dataset.chartState = "unavailable";
      chart.replaceChildren(Object.assign(document.createElement("p"),{className:"chart-fallback",textContent:"Chart unavailable. The exact data table remains available below."}));
      view = null;
    });
  }

  form.addEventListener("submit", event => event.preventDefault());
  form.addEventListener("input",applyFilters);
  form.addEventListener("change",applyFilters);
  form.addEventListener("reset",() => setTimeout(applyFilters,0));
  for (const card of document.querySelectorAll("[data-outcome]")) {
    card.addEventListener("click",() => {outcome.value=card.dataset.outcome;applyFilters();});
  }
  if (window.ResizeObserver) new ResizeObserver(() => applyFilters()).observe(chart);
  applyFilters();
})();
