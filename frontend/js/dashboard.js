/**
 * Dashboard Rendering Logic for AI Career Copilot.
 * Handles SVG Threat Gauge, Defensible Skills, Career Pivots, and 30-60-90 Days Timeline.
 */

export const dashboard = {
    /**
     * Render the complete career analysis report into the DOM.
     * @param {Object} data - Analysis response from backend
     */
    render(data) {
        if (!data) return;

        // 1. Render Candidate Overview Header
        this.renderCandidateHeader(data);

        // 2. Render Threat Gauge Meter
        this.renderThreatGauge(data.threat_score, data.threat_level);

        // 3. Render Skill Defense vs Vulnerability
        this.renderSkillsAssessment(data.defensible_skills, data.vulnerable_skills);

        // 4. Render Safe Career Pivots
        this.renderCareerPivots(data.career_pivots);

        // 5. Render 30-60-90 Days Roadmap
        this.renderRoadmap(data.roadmap);
    },

    /**
     * Render Candidate name, current occupation, and O*NET match.
     */
    renderCandidateHeader(data) {
        const titleEl = document.getElementById('candidate-occupation');
        const onetEl = document.getElementById('candidate-onet');
        const confidenceEl = document.getElementById('onet-confidence');

        if (titleEl) {
            titleEl.textContent = data.current_occupation || 'Chuyên viên Công nghệ';
        }

        if (onetEl) {
            onetEl.textContent = data.matched_onet_code ? `Mã O*NET: ${data.matched_onet_code}` : 'O*NET: Chưa ánh xạ';
        }

        if (confidenceEl && data.confidence !== undefined) {
            const confPercent = Math.round(data.confidence * 100);
            confidenceEl.textContent = `Độ tin cậy: ${confPercent}%`;
            confidenceEl.className = confPercent >= 80 ? 'badge badge-success' : 'badge badge-warning';
        }
    },

    /**
     * Render SVG Threat Gauge with animated stroke.
     * @param {number} score - Threat score (0.0 - 1.0 or 0 - 100)
     * @param {string} level - "Low", "Moderate", "High", "Critical"
     */
    renderThreatGauge(score, level) {
        const rawScore = score > 1 ? score : score * 100;
        const percentage = Math.min(100, Math.max(0, Math.round(rawScore)));

        const scoreEl = document.getElementById('threat-score-number');
        const levelEl = document.getElementById('threat-level-badge');
        const circleEl = document.getElementById('threat-gauge-circle');

        if (scoreEl) {
            // Animated counter effect
            this.animateCounter(scoreEl, 0, percentage, 1200);
        }

        // Color coding & badge text
        let color = 'var(--accent-emerald)';
        let levelText = 'Nguy cơ thấp (An toàn)';
        let badgeClass = 'badge badge-success';

        if (percentage >= 70) {
            color = 'var(--accent-rose)';
            levelText = 'Nguy cơ cao (Cần chuyển đổi)';
            badgeClass = 'badge badge-danger';
        } else if (percentage >= 40) {
            color = 'var(--accent-amber)';
            levelText = 'Nguy cơ trung bình (Cần nâng cao kỹ năng)';
            badgeClass = 'badge badge-warning';
        }

        if (levelEl) {
            levelEl.textContent = level || levelText;
            levelEl.className = badgeClass;
        }

        if (circleEl) {
            // Total circumference = 2 * PI * r (r=45 => C ≈ 282.74)
            const circumference = 282.74;
            const offset = circumference - (percentage / 100) * circumference;
            circleEl.style.stroke = color;
            circleEl.style.strokeDasharray = `${circumference}`;
            setTimeout(() => {
                circleEl.style.transition = 'stroke-dashoffset 1.2s cubic-bezier(0.4, 0, 0.2, 1)';
                circleEl.style.strokeDashoffset = `${offset}`;
            }, 100);
        }
    },

    /**
     * Render Defensible (Safe) vs Vulnerable (Automatable) skills pills.
     */
    renderSkillsAssessment(defensible = [], vulnerable = []) {
        const defensibleContainer = document.getElementById('defensible-skills-container');
        const vulnerableContainer = document.getElementById('vulnerable-skills-container');

        if (defensibleContainer) {
            if (!defensible || defensible.length === 0) {
                defensibleContainer.innerHTML = '<span class="text-muted">Chưa ghi nhận kỹ năng phòng thủ đặc thù</span>';
            } else {
                defensibleContainer.innerHTML = defensible.map(skill => `
                    <span class="skill-tag skill-defensible" title="Kỹ năng khó bị AI thay thế">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                        </svg>
                        ${escapeHtml(skill)}
                    </span>
                `).join('');
            }
        }

        if (vulnerableContainer) {
            if (!vulnerable || vulnerable.length === 0) {
                vulnerableContainer.innerHTML = '<span class="text-muted">Không phát hiện rủi ro tự động hóa đáng kể</span>';
            } else {
                vulnerableContainer.innerHTML = vulnerable.map(skill => `
                    <span class="skill-tag skill-vulnerable" title="Kỹ năng có tỷ lệ tự động hóa cao">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <circle cx="12" cy="12" r="10"/>
                            <line x1="12" y1="8" x2="12" y2="12"/>
                            <line x1="12" y1="16" x2="12.01" y2="16"/>
                        </svg>
                        ${escapeHtml(skill)}
                    </span>
                `).join('');
            }
        }
    },

    /**
     * Render Safe Career Pivots cards / table.
     */
    renderCareerPivots(pivots = []) {
        const container = document.getElementById('career-pivots-container');
        if (!container) return;

        if (!pivots || pivots.length === 0) {
            container.innerHTML = '<p class="text-muted">Không tìm thấy hướng chuyển dịch tương đồng trong cơ sở dữ liệu hiện tại.</p>';
            return;
        }

        container.innerHTML = pivots.map((pivot, idx) => {
            const similarity = Math.round((pivot.similarity_score || pivot.similarity || 0.8) * 100);
            const targetRisk = Math.round((pivot.target_threat || pivot.threat_score || 0.25) * 100);
            const skillsList = pivot.overlap_skills || pivot.transferrable_skills || [];

            return `
                <div class="pivot-card card glass-card">
                    <div class="pivot-header">
                        <div class="pivot-rank">#${idx + 1}</div>
                        <div>
                            <h4 class="pivot-title">${escapeHtml(pivot.target_occupation || pivot.title || 'Vị trí Mục tiêu')}</h4>
                            <span class="pivot-code">${escapeHtml(pivot.onet_code || '')}</span>
                        </div>
                        <div class="pivot-match">
                            <span class="badge badge-primary">${similarity}% Tương đồng</span>
                        </div>
                    </div>
                    <div class="pivot-metrics">
                        <div class="pivot-metric-item">
                            <span class="metric-label">Mức rủi ro AI:</span>
                            <span class="metric-val ${targetRisk < 35 ? 'text-success' : 'text-warning'}">${targetRisk}%</span>
                        </div>
                        <div class="pivot-metric-item">
                            <span class="metric-label">Độ sẵn sàng:</span>
                            <span class="metric-val text-primary">${pivot.readiness || 'Cao'}</span>
                        </div>
                    </div>
                    <div class="pivot-skills">
                        <span class="small-title">Kỹ năng chuyển tiếp sẵn có:</span>
                        <div class="pivot-tags">
                            ${skillsList.slice(0, 5).map(s => `<span class="pivot-tag">${escapeHtml(s)}</span>`).join('')}
                            ${skillsList.length > 5 ? `<span class="pivot-tag">+${skillsList.length - 5}</span>` : ''}
                        </div>
                    </div>
                </div>
            `;
        }).join('');
    },

    /**
     * Render 30-60-90 Days Roadmap timeline.
     */
    renderRoadmap(roadmap = {}) {
        const container = document.getElementById('roadmap-timeline-container');
        if (!container) return;

        const phases = [
            {
                key: 'phase_30',
                days: 'Ngày 1 - 30',
                title: 'Nền tảng & Lấp khoảng cách kỹ năng (Foundation)',
                icon: '🌱',
                data: roadmap.phase_30 || roadmap['30_days'] || {
                    focus: 'Củng cố các kỹ năng nền tảng và bổ sung công cụ AI hỗ trợ công việc hiện tại.',
                    actions: ['Khảo sát công cụ AI cho vai trò hiện tại', 'Tham gia khóa học cơ bản về Prompt Engineering và AI workflows']
                }
            },
            {
                key: 'phase_60',
                days: 'Ngày 31 - 60',
                title: 'Nâng cao & Dự án thực hành (Skill Acquisition)',
                icon: '🚀',
                data: roadmap.phase_60 || roadmap['60_days'] || {
                    focus: 'Xây dựng mini-project kết hợp chuyên môn nghiệp vụ với giải pháp tự động hóa.',
                    actions: ['Triển khai 1 bài toán thực tế của doanh nghiệp', 'Tối ưu hóa quy trình làm việc tăng năng suất gấp 2 lần']
                }
            },
            {
                key: 'phase_90',
                days: 'Ngày 61 - 90',
                title: 'Hoàn thiện hồ sơ & Chuyển dịch (Transition & Portfolio)',
                icon: '🎯',
                data: roadmap.phase_90 || roadmap['90_days'] || {
                    focus: 'Cập nhật CV, Portfolio dự án thực chiến và bắt đầu kết nối ứng tuyển các vị trí an toàn trước làn sóng AI.',
                    actions: ['Cập nhật CV với từ khóa kỹ năng phòng thủ cao', 'Kết nối mạng lưới chuyên gia trong nhóm ngành chuyển dịch an toàn']
                }
            }
        ];

        container.innerHTML = phases.map(phase => {
            const focus = typeof phase.data === 'string' ? phase.data : (phase.data.focus || phase.data.summary || '');
            const actions = Array.isArray(phase.data.actions) ? phase.data.actions : [];

            return `
                <div class="timeline-item">
                    <div class="timeline-badge">${phase.icon}</div>
                    <div class="timeline-card card glass-card">
                        <div class="timeline-header">
                            <span class="timeline-days badge badge-primary">${phase.days}</span>
                            <h4 class="timeline-title">${phase.title}</h4>
                        </div>
                        <p class="timeline-focus">${escapeHtml(focus)}</p>
                        ${actions.length > 0 ? `
                            <ul class="timeline-actions">
                                ${actions.map(act => `<li>${escapeHtml(act)}</li>`).join('')}
                            </ul>
                        ` : ''}
                    </div>
                </div>
            `;
        }).join('');
    },

    /**
     * Animate number counting smoothly.
     */
    animateCounter(element, start, end, duration) {
        let startTime = null;
        function step(timestamp) {
            if (!startTime) startTime = timestamp;
            const progress = Math.min((timestamp - startTime) / duration, 1);
            // Ease out cubic
            const easeProgress = 1 - Math.pow(1 - progress, 3);
            const current = Math.floor(start + (end - start) * easeProgress);
            element.textContent = current;
            if (progress < 1) {
                window.requestAnimationFrame(step);
            } else {
                element.textContent = end;
            }
        }
        window.requestAnimationFrame(step);
    }
};

/**
 * Basic HTML escaping helper.
 */
function escapeHtml(text) {
    if (!text) return '';
    return String(text)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}
