document.addEventListener('DOMContentLoaded', () => {
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const fileUpload = document.getElementById('file-upload');
    const uploadedFilesList = document.getElementById('uploaded-files-list');
    const backendStatus = document.getElementById('backend-status');

    const BACKEND_URL = 'http://127.0.0.1:8000'; // Replace with your backend URL if different

    // Function to add messages to the chat interface
    function addMessage(sender, message, isHtml = false) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', sender === 'user' ? 'user-message' : 'bot-message');
        if (isHtml) {
            messageDiv.innerHTML = message;
        } else {
            messageDiv.textContent = message;
        }
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight; // Scroll to the bottom
    }
    function renderArxivPapers(papers) {
        if (!papers || papers.length === 0) {
            return "<p>No recent papers found for your query.</p>";
        }

        let html = "<h4>Recent Research Papers:</h4><div class='arxiv-results'>";
        papers.forEach(paper => {
            html += `
                <div class="arxiv-paper-card">
                    <h5><a href="${paper.url}" target="_blank" rel="noopener noreferrer">${paper.title}</a></h5>
                    <p><strong>Authors:</strong> ${paper.authors ? paper.authors.join(', ') : 'N/A'}</p>
                    <p><strong>Published:</strong> ${paper.published ? new Date(paper.published).toLocaleDateString() : 'N/A'}</p>
                    <p class="summary">${paper.summary ? paper.summary.substring(0, 200) + '...' : 'No summary available.'}</p>
                </div>
            `;
        });
        html += "</div>";
        return html;
    }
    // Function to update backend status indicator
    async function checkBackendStatus() {
        try {
            const response = await fetch(`${BACKEND_URL}/`);
            if (response.ok) {
                backendStatus.textContent = 'Connected';
                backendStatus.classList.remove('disconnected');
                backendStatus.classList.add('connected');
                // Fetch and display already uploaded documents
                fetchUploadedDocuments();
            } else {
                throw new Error('Backend not reachable');
            }
        } catch (error) {
            console.error('Backend connection error:', error);
            backendStatus.textContent = 'Disconnected';
            backendStatus.classList.remove('connected');
            backendStatus.classList.add('disconnected');
            setTimeout(checkBackendStatus, 5000); // Retry after 5 seconds
        }
    }

    // Function to fetch and display uploaded documents
    async function fetchUploadedDocuments() {
        try {
            const response = await fetch(`${BACKEND_URL}/documents/`);
            if (response.ok) {
                const documents = await response.json();
                uploadedFilesList.innerHTML = ''; // Clear existing list
                if (documents.length > 0) {
                    documents.forEach(doc => {
                        const listItem = document.createElement('li');
                        listItem.textContent = doc.filename;
                        uploadedFilesList.appendChild(listItem);
                    });
                } else {
                    const listItem = document.createElement('li');
                    listItem.textContent = 'No PDFs uploaded yet.';
                    uploadedFilesList.appendChild(listItem);
                }
            } else {
                console.error('Failed to fetch uploaded documents:', response.statusText);
                addMessage('bot', 'Failed to load previously uploaded documents.');
            }
        } catch (error) {
            console.error('Error fetching uploaded documents:', error);
            addMessage('bot', 'Error connecting to backend to fetch documents.');
        }
    }

    // Call checkBackendStatus on page load
    checkBackendStatus();

    // Handle file upload
    fileUpload.addEventListener('change', async (event) => {
        const files = event.target.files;
        if (files.length === 0) return;

        addMessage('user', `Uploading ${files.length} file(s)...`);

        for (const file of files) {
            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await fetch(`${BACKEND_URL}/upload-document/`, {
                    method: 'POST',
                    body: formData,
                });

                if (response.ok) {
                    const result = await response.json();
                    addMessage('bot', `Successfully processed "${result.filename}"! You can now ask questions about it.`);
                    // Update the list of uploaded files
                    fetchUploadedDocuments(); 
                } else {
                    const errorData = await response.json();
                    addMessage('bot', `Failed to upload "${file.name}": ${errorData.detail || response.statusText}`);
                }
            } catch (error) {
                console.error('Error uploading file:', error);
                addMessage('bot', `An error occurred during upload for "${file.name}". Please check the backend connection.`);
            }
        }
        fileUpload.value = ''; // Clear the file input
    });

    // Simple NLP-like logic to extract filename from user query
    function extractFilename(query) {
        // Regex to find common PDF filename patterns (e.g., "document.pdf", "report 2023.pdf")
        const pdfPattern = /(\b\w+\.pdf\b|\b\w+\s\w+\.pdf\b|\b\w+-\w+\.pdf\b)/gi;
        const matches = query.match(pdfPattern);
        
        if (matches && matches.length > 0) {
            // Take the first detected filename
            return matches[0].toLowerCase(); // Convert to lowercase for consistent matching
        }
        return null;
    }

    // Handle sending message
    async function sendMessage() {
        const question = userInput.value.trim();
        if (question === '') return;

        addMessage('user', question);
        userInput.value = ''; // Clear input field

        const thinkingMessageDiv = document.createElement('div');
        thinkingMessageDiv.classList.add('message', 'bot-message', 'thinking-message');
        thinkingMessageDiv.innerHTML = 'Thinking... <span class="dot-flashing"></span>'; // Added simple loading animation
        chatMessages.appendChild(thinkingMessageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight; // Scroll to the bottom

        const filename = extractFilename(question);
        
        let queryBody = { question: question };
        if (filename) {
            const cleanedQuestion = question.replace(filename, '').trim();
            queryBody.question = cleanedQuestion || question;
            queryBody.filename = filename;
            addMessage('bot', `Searching specifically in <strong>${filename}</strong>...`, true);
        }

        try {
            const response = await fetch(`${BACKEND_URL}/query-document/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(queryBody),
            });

            const result = await response.json();
            
            // Remove the "Thinking..." message
            chatMessages.removeChild(thinkingMessageDiv);

            if (response.ok) {
                if (result.type === 'arxiv_papers') {
                    const arxivHtml = renderArxivPapers(result.answer); // result.answer is now the array of papers
                    addMessage('bot', arxivHtml, true); // Pass as HTML
                } else {
                    // For general text responses (including RAG answers or LLM synthesis)
                    // Consider using a markdown parser here if you want bolding, lists etc. to render.
                    // If you added marked.js: addMessage('bot', renderMarkdown(result.answer), true);
                    addMessage('bot', result.answer); // Otherwise, display as plain text
                }
            } else {
                addMessage('bot', `Error: ${result.detail || response.statusText}`);
            }
        } catch (error) {
            console.error('Error sending message:', error);
            // Remove thinking message if error occurs
            chatMessages.removeChild(thinkingMessageDiv);
            addMessage('bot', `An error occurred: Could not connect to the backend or process request.`);
        }
    }

    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            sendMessage();
        }
    });

    // Initial check for backend status and uploaded documents
    checkBackendStatus();
});

