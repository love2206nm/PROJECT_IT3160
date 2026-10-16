/**
 * Mermaid Diagram Renderer for Career Transition & State Graphs.
 */

export const mermaidRenderer = {
    initialized: false,

    async init() {
        if (this.initialized) return;
        if (window.mermaid) {
            window.mermaid.initialize({
                startOnLoad: false,
                theme: 'dark',
                themeVariables: {
                    darkMode: true,
                    background: '#090d16',
                    primaryColor: '#3b82f6',
                    primaryTextColor: '#f8fafc',
                    primaryBorderColor: '#60a5fa',
                    lineColor: '#94a3b8',
                    secondaryColor: '#1e293b',
                    tertiaryColor: '#0f172a'
                }
            });
            this.initialized = true;
        }
    },

    /**
     * Render a mermaid graph definition into a DOM element.
     * @param {string} containerId - Target HTML element ID
     * @param {string} graphDefinition - Mermaid diagram syntax
     */
    async render(containerId, graphDefinition) {
        await this.init();
        const container = document.getElementById(containerId);
        if (!container || !window.mermaid) return;

        try {
            const uniqueId = `mermaid-svg-${Date.now()}`;
            const { svg } = await window.mermaid.render(uniqueId, graphDefinition);
            container.innerHTML = svg;
        } catch (err) {
            console.error('Failed to render Mermaid chart:', err);
            container.innerHTML = `<p class="text-muted">Không thể tạo sơ đồ trực quan: ${err.message}</p>`;
        }
    }
};
