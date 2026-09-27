import { Fragment, useEffect, useRef, useState, type FormEvent, type ReactNode } from 'react';
import { MessageCircle, Send, Trash2, X } from 'lucide-react';
import { sendChat, type ChatResponse } from '../services/api';

type Message = { role: 'assistant' | 'user'; answer: string; sources?: ChatResponse['sources'] };
const suggestions = ['What is the difference between NSCLC and SCLC?', 'How is lung cancer diagnosed?', 'What does Grad-CAM show?'];

function inlineFormatting(text: string): ReactNode[] {
  return text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).filter(Boolean).map((part, index) => {
    if (part.startsWith('**') && part.endsWith('**')) return <strong key={index}>{part.slice(2, -2)}</strong>;
    if (part.startsWith('`') && part.endsWith('`')) return <code key={index}>{part.slice(1, -1)}</code>;
    return <Fragment key={index}>{part}</Fragment>;
  });
}

function renderAnswer(answer: string) {
  const blocks: ReactNode[] = [];
  const lines = answer.trim().split(/\r?\n/);
  let paragraph: string[] = [];
  let index = 0;
  const flushParagraph = () => {
    if (paragraph.length) {
      blocks.push(<p key={`p-${index++}`}>{inlineFormatting(paragraph.join(' '))}</p>);
      paragraph = [];
    }
  };

  while (index < lines.length) {
    const line = lines[index].trim();
    if (!line) { flushParagraph(); index += 1; continue; }
    const heading = line.match(/^#{1,3}\s+(.+)$/);
    if (heading) {
      flushParagraph();
      blocks.push(<h4 key={`h-${index++}`}>{inlineFormatting(heading[1])}</h4>);
      continue;
    }
    const listItem = line.match(/^(?:[-*•])\s+(.+)$/);
    const orderedItem = line.match(/^\d+[.)]\s+(.+)$/);
    if (listItem || orderedItem) {
      flushParagraph();
      const ordered = Boolean(orderedItem);
      const items: string[] = [];
      while (index < lines.length) {
        const current = lines[index].trim();
        const match = ordered ? current.match(/^\d+[.)]\s+(.+)$/) : current.match(/^(?:[-*•])\s+(.+)$/);
        if (!match) break;
        items.push(match[1]);
        index += 1;
      }
      const List = ordered ? 'ol' : 'ul';
      blocks.push(<List key={`list-${index}`}>{items.map((item, itemIndex) => <li key={itemIndex}>{inlineFormatting(item)}</li>)}</List>);
      continue;
    }
    paragraph.push(line);
    index += 1;
  }
  flushParagraph();
  return blocks;
}

export default function ChatWidget({ context }: { context?: Record<string, unknown> }) {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([{ role: 'assistant', answer: 'Hi, I’m the Lung Cancer AI Assistant. Ask me about lung cancer or how this research application works.' }]);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const conversationId = useRef(crypto.randomUUID());
  const messageListRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const list = messageListRef.current;
    if (list) list.scrollTo({ top: list.scrollHeight, behavior: 'smooth' });
  }, [messages, busy, error]);
  const send = async (event?: FormEvent, suggested?: string) => {
    event?.preventDefault();
    const message = (suggested ?? input).trim();
    if (!message || busy) return;
    setMessages(previous => [...previous, { role: 'user', answer: message }]);
    setInput(''); setBusy(true); setError('');
    try {
      const history = messages.slice(-8).map(({ role, answer }) => ({ role, content: answer }));
      const result = await sendChat(message, conversationId.current, context, history);
      setMessages(previous => [...previous, { role: 'assistant', answer: result.answer, sources: result.sources }]);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : 'The assistant is unavailable right now.');
    } finally { setBusy(false); }
  };
  const clear = () => {
    conversationId.current = crypto.randomUUID();
    setMessages([{ role: 'assistant', answer: 'Conversation cleared. What would you like to know?' }]);
    setError('');
  };
  return <div className="chat-widget">
    {open && <section className="chat-panel" aria-label="Lung Cancer AI Assistant">
      <header className="chat-header"><span className="chat-mark">🫁</span><div><strong>Lung Cancer AI Assistant</strong><small>Educational research assistant</small></div><button type="button" aria-label="Clear conversation" onClick={clear}><Trash2 size={16}/></button><button type="button" aria-label="Close chat" onClick={() => setOpen(false)}><X size={18}/></button></header>
      <div className="chat-messages" aria-live="polite" ref={messageListRef}>
        {messages.map((message, index) => <article className={`chat-message ${message.role}`} key={index}>
          {message.role === 'assistant' && <span className="chat-avatar" aria-hidden="true">🫁</span>}
          <div className="chat-message-body">
            <span className="chat-byline">{message.role === 'assistant' ? 'Lung Cancer AI Assistant' : 'You'}</span>
            <div className="chat-answer">{renderAnswer(message.answer)}</div>
            {message.sources?.length ? <div className="chat-sources"><b>Trusted sources</b>{message.sources.map(source => <a key={source.url} href={source.url} target="_blank" rel="noreferrer">{source.title}<span aria-hidden="true"> ↗</span></a>)}</div> : null}
          </div>
        </article>)}
        {busy && <div className="chat-typing" role="status"><span className="typing-dots" aria-hidden="true"><i/><i/><i/></span>Putting together a helpful answer…</div>}
        {error && <p className="error" role="alert">{error}</p>}
        {messages.length === 1 && <div className="chat-suggestions">{suggestions.map(question => <button type="button" key={question} onClick={() => void send(undefined, question)}>{question}</button>)}</div>}
      </div>
      <form className="chat-form" onSubmit={event => void send(event)}><input value={input} onChange={event => setInput(event.target.value)} maxLength={2000} placeholder="Ask about lung cancer or this app…" aria-label="Your question"/><button type="submit" disabled={busy || !input.trim()} aria-label="Send message"><Send size={17}/></button></form>
      <p className="chat-disclaimer">Educational only. Not a diagnosis or emergency service.</p>
    </section>}
    <button className="chat-launcher" type="button" aria-label={open ? 'Close assistant' : 'Open Lung Cancer AI Assistant'} onClick={() => setOpen(value => !value)}>{open ? <X/> : <MessageCircle/>}</button>
  </div>;
}
