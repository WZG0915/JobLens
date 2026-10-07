const state = { sessionId: null, selectedFile: null, busy: false };

const messages = document.querySelector('#messages');
const form = document.querySelector('#chat-form');
const input = document.querySelector('#message-input');
const fileInput = document.querySelector('#resume-input');
const fileChip = document.querySelector('#file-chip');
const fileName = document.querySelector('#file-name');
const sendButton = document.querySelector('#send-button');
const clearButton = document.querySelector('#clear-button');
const topK = document.querySelector('#top-k');
const template = document.querySelector('#recommendation-template');

function escapeText(value) {
  return String(value ?? '');
}

function appendMessage(role, text, error = false) {
  const article = document.createElement('article');
  article.className = `message ${role === 'user' ? 'user-message' : 'assistant-message'}`;
  if (role !== 'user') {
    const avatar = document.createElement('div');
    avatar.className = 'avatar'; avatar.textContent = 'J';
    article.appendChild(avatar);
  }
  const bubble = document.createElement('div');
  bubble.className = `bubble${error ? ' error-bubble' : ''}`;
  const paragraph = document.createElement('p');
  paragraph.textContent = escapeText(text);
  bubble.appendChild(paragraph); article.appendChild(bubble); messages.appendChild(article);
  article.scrollIntoView({ behavior: 'smooth', block: 'end' });
  return article;
}

function appendTyping() {
  const article = document.createElement('article');
  article.className = 'message assistant-message';
  article.innerHTML = '<div class="avatar">J</div><div class="bubble"><span class="typing"><i></i><i></i><i></i></span></div>';
  messages.appendChild(article); article.scrollIntoView({ behavior: 'smooth' });
  return article;
}

function fillList(element, values, emptyText) {
  element.replaceChildren();
  const content = values?.length ? values : [emptyText];
  content.forEach(value => {
    const item = document.createElement('li'); item.textContent = value; element.appendChild(item);
  });
}

function appendRecommendations(items) {
  if (!items?.length) return;
  const stack = document.createElement('section'); stack.className = 'recommendation-stack';
  items.forEach(job => {
    const card = template.content.firstElementChild.cloneNode(true);
    card.querySelector('.job-rank').append(String(job.rank));
    card.querySelector('.job-company').textContent = job.company || '未知公司';
    card.querySelector('.job-title').textContent = job.title;
    card.querySelector('.score-ring strong').textContent = job.final_score.toFixed(1);
    card.querySelector('.rag-score').textContent = `${(job.retrieval_score * 100).toFixed(1)}%`;
    card.querySelector('.evidence-score').textContent = job.evidence_score.toFixed(1);
    card.querySelector('.coverage-score').textContent = `${(job.evidence_coverage * 100).toFixed(0)}%`;
    const hard = card.querySelector('.hard-fail-score'); hard.textContent = String(job.hard_constraints_failed);
    if (job.hard_constraints_failed) hard.classList.add('alert');
    const tags = card.querySelector('.skill-tags');
    (job.matched_skills || []).slice(0, 10).forEach(skill => {
      const tag = document.createElement('span'); tag.textContent = skill; tags.appendChild(tag);
    });
    fillList(card.querySelector('.strength-list'), job.strengths, '暂未找到明确优势证据');
    fillList(card.querySelector('.gap-list'), job.gaps, '未发现明显能力差距');
    fillList(card.querySelector('.suggestion-list'), job.suggestions, '继续补充量化成果与项目证据');
    card.querySelector('.job-source').textContent = `岗位数据：${job.job_source}`;
    stack.appendChild(card);
  });
  messages.appendChild(stack); stack.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function setBusy(value) {
  state.busy = value; sendButton.disabled = value; input.disabled = value;
  sendButton.querySelector('span:first-child').textContent = value ? '处理中' : '发送';
}

function updateFile(file) {
  state.selectedFile = file;
  if (file) { fileName.textContent = file.name; fileChip.classList.remove('hidden'); }
  else { fileInput.value = ''; fileChip.classList.add('hidden'); fileName.textContent = ''; }
}

async function sendMessage(message) {
  if (state.busy) return;
  if (!state.sessionId && !state.selectedFile) {
    appendMessage('assistant', '请先点击输入框左侧的“＋”上传PDF简历。', true); return;
  }
  const text = message.trim() || (state.selectedFile ? '解析我的简历' : '推荐5个岗位');
  appendMessage('user', text); input.value = ''; input.style.height = 'auto';
  const typing = appendTyping(); setBusy(true);
  const payload = new FormData(); payload.append('message', text); payload.append('top_k', topK.value);
  if (state.sessionId) payload.append('session_id', state.sessionId);
  if (state.selectedFile) payload.append('resume_file', state.selectedFile);
  try {
    const response = await fetch('/api/chat', { method: 'POST', body: payload });
    const data = await response.json();
    typing.remove();
    if (!response.ok) throw new Error(data.detail || '本地服务处理失败');
    state.sessionId = data.session_id; clearButton.disabled = false; updateFile(null);
    appendMessage('assistant', data.message);
    appendRecommendations(data.recommendations);
    if (data.answered_by === 'ai' && data.tool_calls?.length) {
      const uniqueTools = [...new Set(data.tool_calls)];
      appendMessage('assistant', `AI 已调用本地工具：${uniqueTools.join('、')}`);
    }
    if (data.warnings?.length) appendMessage('assistant', `提示：${data.warnings.join('；')}`);
  } catch (error) {
    typing.remove(); appendMessage('assistant', error.message || '请求失败，请检查本地服务。', true);
  } finally { setBusy(false); input.focus(); }
}

form.addEventListener('submit', event => { event.preventDefault(); sendMessage(input.value); });
input.addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); form.requestSubmit(); }
});
input.addEventListener('input', () => {
  input.style.height = 'auto'; input.style.height = `${Math.min(input.scrollHeight, 120)}px`;
});
fileInput.addEventListener('change', () => updateFile(fileInput.files[0] || null));
document.querySelector('#remove-file').addEventListener('click', () => updateFile(null));
document.querySelectorAll('[data-prompt]').forEach(button => {
  button.addEventListener('click', () => { input.value = button.dataset.prompt; input.focus(); });
});
clearButton.addEventListener('click', async () => {
  if (state.sessionId) await fetch(`/api/sessions/${state.sessionId}`, { method: 'DELETE' });
  state.sessionId = null; updateFile(null); clearButton.disabled = true;
  appendMessage('assistant', '本地简历会话已经清除。你可以重新上传另一份PDF。');
});

fetch('/api/health').then(response => response.json()).then(data => {
  const dot = document.querySelector('#status-dot');
  dot.classList.add(data.status === 'ready' ? 'ready' : 'error');
  if (data.ai_ready && data.index_ready && data.jobs_ready) {
    document.querySelector('#status-text').textContent = 'AI 对话与岗位索引已就绪';
  } else if (!data.ai_ready) {
    document.querySelector('#status-text').textContent = 'AI 未连接，将使用本地模式';
  } else {
    document.querySelector('#status-text').textContent = '需要先构建岗位索引';
  }
  const aiName = data.ai_model ? ` · AI：${data.ai_model}` : '';
  document.querySelector('#model-name').textContent = `BGE：${data.model_name}${aiName}`;
}).catch(() => {
  document.querySelector('#status-dot').classList.add('error');
  document.querySelector('#status-text').textContent = '无法连接本地服务';
});
