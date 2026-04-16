document.addEventListener("DOMContentLoaded", () => {
    const queryInput = document.getElementById("queryInput");
    const loader = document.getElementById("loader");
    const emptyState = document.getElementById("emptyState");
    const resultCard = document.getElementById("resultCard");
    const searchContainer = document.querySelector(".search-container");

    const statusIcon = document.getElementById("statusIcon");
    const domainTitle = document.getElementById("domainTitle");
    const confidenceValue = document.getElementById("confidenceValue");
    const confidenceBar = document.getElementById("confidenceBar");
    const contentType = document.getElementById("contentType");
    const extractedTitle = document.getElementById("extractedTitle");
    const pipelineDecision = document.getElementById("pipelineDecision");
    const reasonsContainer = document.getElementById("reasonsContainer");

    const icons = {
        video: `<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="5" width="16" height="14" rx="2"/><line x1="22" y1="9" x2="18" y2="11"/><line x1="18" y1="13" x2="22" y2="15"/></svg>`,
        trash: `<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/></svg>`,
        uncertain: `<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 015.83 1c0 2-3 3-3 3M12 17h.01"/></svg>`
    };

    let debounceTimer;

    queryInput.addEventListener("input", (e) => {
        const query = e.target.value.trim();
        
        clearTimeout(debounceTimer);
        
        if (query.length === 0) {
            hideResults();
            return;
        }

        debounceTimer = setTimeout(() => {
            fetchLabel(query);
        }, 400); // 400ms debounce
    });

    async function fetchLabel(query) {
        showLoader();
        try {
            const res = await fetch("/label", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query_text: query })
            });
            if (!res.ok) throw new Error("API Error");
            const data = await res.json();
            renderResults(data);
        } catch (err) {
            console.error(err);
            hideLoader();
        }
    }

    function showLoader() {
        loader.classList.remove("hidden");
    }

    function hideLoader() {
        loader.classList.add("hidden");
    }

    function hideResults() {
        hideLoader();
        resultCard.classList.add("hidden");
        emptyState.classList.remove("hidden");
        searchContainer.classList.remove("active-search");
    }

    function renderResults(data) {
        hideLoader();
        emptyState.classList.add("hidden");
        searchContainer.classList.add("active-search");
        resultCard.classList.remove("hidden");

        const percent = Math.round(data.confidence * 100);
        confidenceValue.textContent = `${percent}%`;
        
        // Wait minor delay to trigger CSS animation smoothly for bar
        setTimeout(() => {
            confidenceBar.style.width = `${percent}%`;
        }, 50);

        // Map domains
        const isVideo = data.domain_label === "prof_video";
        const isTrash = data.domain_label === "non_video";

        if (isVideo || data.is_prof_video) {
            statusIcon.innerHTML = icons.video;
            statusIcon.className = "icon-box success";
            domainTitle.textContent = "Video Content";
            domainTitle.style.color = "var(--success-neon)";
        } else if (isTrash) {
            statusIcon.innerHTML = icons.trash;
            statusIcon.className = "icon-box trash";
            domainTitle.textContent = "Unrelated (Trash)";
            domainTitle.style.color = "var(--text-secondary)";
        } else {
            statusIcon.innerHTML = icons.uncertain;
            statusIcon.className = "icon-box";
            domainTitle.textContent = "Uncertain";
            domainTitle.style.color = "var(--accent-purple)";
        }

        contentType.textContent = data.content_type || "N/A";
        extractedTitle.textContent = data.title || "No exact match";
        
        const decisionText = data.decision.replace("_", " ").toUpperCase();
        pipelineDecision.textContent = decisionText;

        // Render reasons as small tags
        reasonsContainer.innerHTML = "";
        data.reasons.forEach(r => {
            const span = document.createElement("span");
            span.className = "reason-tag";
            span.textContent = r;
            reasonsContainer.appendChild(span);
        });
    }
});
