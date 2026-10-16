/**
 * Pure Chatbot API Client for AI Career Copilot.
 * Connects to FastAPI SSE streaming endpoint with support for both
 * conversational text messages and direct CV file attachments.
 */

export const api = {
    /**
     * Streams real-time token chunks from backend Copilot endpoint.
     * @param {string} sessionId - Active session thread ID
     * @param {string} message - User message text
     * @param {File|null} file - Optional attached CV file (.pdf, .docx, .txt)
     * @param {Function} onToken - Callback for each progressive token
     * @param {Function} onDone - Callback when stream completes
     * @param {Function} onError - Callback for errors
     */
    async streamChat(sessionId, message, file = null, onToken, onDone, onError) {
        try {
            let response;

            if (file) {
                // Multipart FormData submission when file is attached
                const formData = new FormData();
                formData.append('session_id', sessionId);
                formData.append('message', message || '');
                formData.append('file', file);

                response = await fetch('/api/v1/copilot/chat/stream', {
                    method: 'POST',
                    body: formData,
                });
            } else {
                // Fast JSON submission for standard conversational Q&A
                response = await fetch('/api/v1/copilot/chat/stream', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        session_id: sessionId,
                        message: message || '',
                    }),
                });
            }

            if (!response.ok) {
                const err = await response.json().catch(() => ({ detail: 'Lỗi máy chủ khi kết nối Copilot' }));
                throw new Error(err.detail || `Lỗi máy chủ (${response.status})`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder('utf-8');
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop(); // Hold back incomplete line

                for (const line of lines) {
                    const trimmed = line.trim();
                    if (!trimmed || !trimmed.startsWith('data:')) continue;

                    const dataContent = trimmed.substring(5).trim();
                    if (dataContent === '[DONE]') {
                        if (onDone) onDone();
                        return;
                    }

                    try {
                        const parsed = JSON.parse(dataContent);
                        if (parsed.token && onToken) {
                            onToken(parsed.token);
                        }
                        if (parsed.error && onError) {
                            onError(new Error(parsed.error));
                            return;
                        }
                    } catch (parseErr) {
                        console.warn('Malformed SSE event chunk:', dataContent, parseErr);
                    }
                }
            }

            if (onDone) onDone();
        } catch (err) {
            if (onError) onError(err);
            else console.error('Streaming error:', err);
        }
    }
};
