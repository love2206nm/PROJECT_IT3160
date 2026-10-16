/**
 * Pure Chatbot Orchestrator for AI Career Copilot.
 * Manages conversational state, file attachment, real-time SSE streaming,
 * and Markdown formatting in a single, unified ChatGPT-style interface.
 */

import { api } from './api.js';

class ChatApp {
    constructor() {
        this.sessionId = this.getOrCreateSessionId();
        this.selectedFile = null;
        this.isGenerating = false;

        this.initElements();
        this.bindEvents();
    }

    getOrCreateSessionId() {
        let sid = sessionStorage.getItem('career_copilot_session');
        if (!sid) {
            sid = typeof crypto !== 'undefined' && crypto.randomUUID
                ? crypto.randomUUID()
                : 'sess_' + Date.now() + '_' + Math.random().toString(36).substring(2, 9);
            sessionStorage.setItem('career_copilot_session', sid);
        }
        return sid;
    }

    initElements() {
        this.messagesContainer = document.getElementById('chat-messages');
        this.inputDock = document.getElementById('chat-input-dock');
        this.inputText = document.getElementById('chat-input-text');
        this.btnSend = document.getElementById('btn-send');
        this.btnAttach = document.getElementById('btn-attach');
        this.fileInput = document.getElementById('file-attachment');
        this.attachmentPreview = document.getElementById('attachment-preview');
        this.attachedFilename = document.getElementById('attached-filename');
        this.btnRemoveAttachment = document.getElementById('btn-remove-attachment');
        this.btnNewChat = document.getElementById('btn-new-chat');
    }

    bindEvents() {
        // 1. Textarea Auto-expand & Enter to submit
        if (this.inputText) {
            this.inputText.addEventListener('input', () => this.autoExpandTextarea());
            this.inputText.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.sendMessage();
                }
            });
        }

        // 2. Send button
        if (this.btnSend) {
            this.btnSend.addEventListener('click', () => this.sendMessage());
        }

        // 3. Attach file button
        if (this.btnAttach && this.fileInput) {
            this.btnAttach.addEventListener('click', () => this.fileInput.click());
            this.fileInput.addEventListener('change', (e) => this.handleFileSelected(e.target.files[0]));
        }

        // 4. Remove attached file
        if (this.btnRemoveAttachment) {
            this.btnRemoveAttachment.addEventListener('click', () => this.clearAttachment());
        }

        // 5. New chat button
        if (this.btnNewChat) {
            this.btnNewChat.addEventListener('click', () => this.startNewChat());
        }
    }

    autoExpandTextarea() {
        if (!this.inputText) return;
        this.inputText.style.height = 'auto';
        this.inputText.style.height = Math.min(this.inputText.scrollHeight, 160) + 'px';
    }

    handleFileSelected(file) {
        if (!file) return;
        this.selectedFile = file;

        if (this.attachedFilename && this.attachmentPreview) {
            this.attachedFilename.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
            this.attachmentPreview.classList.remove('hidden');
        }

        if (this.inputText) {
            this.inputText.focus();
        }
    }

    clearAttachment() {
        this.selectedFile = null;
        if (this.fileInput) this.fileInput.value = '';
        if (this.attachmentPreview) this.attachmentPreview.classList.add('hidden');
        if (this.attachedFilename) this.attachedFilename.textContent = '';
    }

    startNewChat() {
        if (this.isGenerating) return;
        this.sessionId = typeof crypto !== 'undefined' && crypto.randomUUID
            ? crypto.randomUUID()
            : 'sess_' + Date.now();
        sessionStorage.setItem('career_copilot_session', this.sessionId);
        this.clearAttachment();

        if (this.messagesContainer) {
            this.messagesContainer.innerHTML = `
                <div class="chat-bubble chat-assistant">
                    <div class="chat-avatar">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M12 2a2 2 0 0 1 2 2v2a2 2 0 0 1-2 2 2 2 0 0 1-2-2V4a2 2 0 0 1 2-2z"/>
                            <rect x="4" y="8" width="16" height="12" rx="2"/>
                        </svg>
                    </div>
                    <div class="chat-content">
                        <p>Đã khởi tạo phiên tư vấn mới!</p>
                        <p>Bạn có thể gõ câu hỏi hoặc nhấn nút chiếc ghim (📎) bên dưới để gửi CV của mình nhé.</p>
                    </div>
                </div>
            `;
        }
        if (this.inputText) {
            this.inputText.value = '';
            this.inputText.style.height = 'auto';
            this.inputText.focus();
        }
    }

    async sendMessage() {
        if (this.isGenerating) return;

        const text = this.inputText ? this.inputText.value.trim() : '';
        const file = this.selectedFile;

        if (!text && !file) return;

        this.isGenerating = true;
        if (this.btnSend) this.btnSend.disabled = true;

        // 1. Append User message bubble
        this.appendUserMessage(text, file);

        // 2. Clear input & attachment
        if (this.inputText) {
            this.inputText.value = '';
            this.inputText.style.height = 'auto';
        }
        this.clearAttachment();

        // 3. Append Assistant placeholder bubble with blinking cursor
        const assistantBubble = this.appendAssistantMessage('');
        assistantBubble.classList.add('typing-cursor');

        let fullOutput = '';

        // 4. Stream real-time tokens via SSE
        await api.streamChat(
            this.sessionId,
            text,
            file,
            (token) => {
                fullOutput += token;
                assistantBubble.innerHTML = this.formatMarkdown(fullOutput);
                this.scrollToBottom();
            },
            () => {
                assistantBubble.classList.remove('typing-cursor');
                assistantBubble.innerHTML = this.formatMarkdown(fullOutput);
                this.isGenerating = false;
                if (this.btnSend) this.btnSend.disabled = false;
                this.scrollToBottom();
                if (this.inputText) this.inputText.focus();
            },
            (err) => {
                assistantBubble.classList.remove('typing-cursor');
                assistantBubble.innerHTML = `<span style="color:var(--accent-rose);">Lỗi phản hồi: ${this.escapeHtml(err.message)}</span>`;
                this.isGenerating = false;
                if (this.btnSend) this.btnSend.disabled = false;
                this.scrollToBottom();
            }
        );
    }

    appendUserMessage(text, file = null) {
        if (!this.messagesContainer) return;

        const bubble = document.createElement('div');
        bubble.className = 'chat-bubble chat-user';

        const avatar = document.createElement('div');
        avatar.className = 'chat-avatar';
        avatar.innerHTML = `
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
                <circle cx="12" cy="7" r="4"/>
            </svg>
        `;

        const content = document.createElement('div');
        content.className = 'chat-content';

        let innerHtml = '';
        if (file) {
            innerHtml += `
                <div class="user-attachment-badge">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                        <polyline points="14 2 14 8 20 8"/>
                    </svg>
                    <span>${this.escapeHtml(file.name)}</span>
                </div>
            `;
        }

        if (text) {
            innerHtml += `<p>${this.escapeHtml(text).replace(/\n/g, '<br/>')}</p>`;
        } else if (file) {
            innerHtml += '<p><em>Đã gửi file CV để phân tích nghề nghiệp.</em></p>';
        }

        content.innerHTML = innerHtml;
        bubble.appendChild(avatar);
        bubble.appendChild(content);

        this.messagesContainer.appendChild(bubble);
        this.scrollToBottom();
    }

    appendAssistantMessage(initialText = '') {
        if (!this.messagesContainer) return null;

        const bubble = document.createElement('div');
        bubble.className = 'chat-bubble chat-assistant';

        const avatar = document.createElement('div');
        avatar.className = 'chat-avatar';
        avatar.innerHTML = `
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M12 2a2 2 0 0 1 2 2v2a2 2 0 0 1-2 2 2 2 0 0 1-2-2V4a2 2 0 0 1 2-2z"/>
                <rect x="4" y="8" width="16" height="12" rx="2"/>
            </svg>
        `;

        const content = document.createElement('div');
        content.className = 'chat-content';
        content.innerHTML = this.formatMarkdown(initialText);

        bubble.appendChild(avatar);
        bubble.appendChild(content);

        this.messagesContainer.appendChild(bubble);
        this.scrollToBottom();

        return content;
    }

    scrollToBottom() {
        if (this.messagesContainer) {
            this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
        }
    }

    escapeHtml(str) {
        if (!str) return '';
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    renderTables(text) {
        if (!text || !text.includes('|')) return text;
        const tableRegex = /((?:^[ \t]*\|[^\n]+\|[ \t]*\r?\n)+(?:^[ \t]*\|[ \t]*:?[-]+:?[ \t]*(?:\|[ \t]*:?[-]+:?[ \t]*)+\|[ \t]*\r?\n)(?:^[ \t]*\|[^\n]+\|[ \t]*(?:\r?\n|$))*)/gm;
        return text.replace(tableRegex, (match) => {
            const lines = match.trim().split(/\r?\n/).map(l => l.trim()).filter(Boolean);
            if (lines.length < 2) return match;

            const parseRow = (line) => {
                const trimmed = line.replace(/^\|/, '').replace(/\|$/, '');
                return trimmed.split('|').map(c => c.trim());
            };

            const headerCells = parseRow(lines[0]);
            const delimiterRow = lines[1];
            if (!/^\|?[ \t]*:?[-]+:?[ \t]*(?:\|[ \t]*:?[-]+:?[ \t]*)+\|?$/.test(delimiterRow)) {
                return match;
            }

            let html = '<div class="chat-table-wrapper"><table class="chat-markdown-table">';
            html += '<thead><tr>';
            for (const cell of headerCells) {
                html += `<th>${cell.replace(/\*\*(.*?)\*\*/g, '$1')}</th>`;
            }
            html += '</tr></thead><tbody>';

            for (let i = 2; i < lines.length; i++) {
                const cells = parseRow(lines[i]);
                html += '<tr>';
                for (let j = 0; j < headerCells.length; j++) {
                    const cellVal = cells[j] !== undefined ? cells[j] : '';
                    const formattedCell = cellVal
                        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                        .replace(/\*(.*?)\*/g, '<em>$1</em>');
                    html += `<td>${formattedCell}</td>`;
                }
                html += '</tr>';
            }
            html += '</tbody></table></div>\n\n';
            return html;
        });
    }

    formatMarkdown(text) {
        if (!text) return '';

        // 1. Clean isolated '#' characters on single lines
        let cleaned = text.replace(/^\s*#\s*$/gm, '');

        // 2. Parse Markdown Tables before newline conversion
        let html = this.renderTables(cleaned);

        // 3. Headers, bold, italics, code, lists
        html = html
            .replace(/^### (.*?)(?:\r?\n|$)/gm, '<h3>$1</h3>')
            .replace(/^## (.*?)(?:\r?\n|$)/gm, '<h3>$1</h3>')
            .replace(/^# (.*?)(?:\r?\n|$)/gm, '<h3>$1</h3>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`([^`]+)`/g, '<code style="background:rgba(255,255,255,0.1);padding:2px 6px;border-radius:4px;font-size:0.9em;">$1</code>')
            .replace(/^---$/gm, '<hr/>')
            .replace(/^\s*[\-\*•]\s+(.*$)/gm, '<li>$1</li>')
            .replace(/^\s*\d+\.\s+(.*$)/gm, '<li>$1</li>');

        // 4. Properly group contiguous list items into single <ul> blocks
        html = html.replace(/((?:<li>[\s\S]*?<\/li>\s*)+)/g, '<ul class="chat-list">$1</ul>');

        // 5. Newlines outside table wrappers and list wrappers
        const parts = html.split(/(<div class="chat-table-wrapper">[\s\S]*?<\/div>|<ul class="chat-list">[\s\S]*?<\/ul>)/g);
        for (let i = 0; i < parts.length; i++) {
            if (!parts[i].startsWith('<div class="chat-table-wrapper">') && !parts[i].startsWith('<ul class="chat-list">')) {
                parts[i] = parts[i]
                    .replace(/\n\n/g, '<br/><br/>')
                    .replace(/\n/g, '<br/>');
            }
        }
        return parts.join('');
    }
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
    window.chatApp = new ChatApp();
});
