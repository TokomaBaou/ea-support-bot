"use client";

import { useEffect, useRef, useState } from "react";
import styles from "./ChatWidget.module.css";

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

type SourceFaq = {
  question: string;
  answer: string;
  category: string;
};

type Rating = "up" | "down";

type Message = {
  id: string;
  role: "user" | "bot";
  text: string;
  serverMessageId?: string;
  sources?: SourceFaq[];
  originalQuery?: string;
  detailRequested?: boolean;
  rating?: Rating;
  isFallback?: boolean;
  matchedSource?: string;
};

type ChatApiResponse = {
  answer: string;
  sources: SourceFaq[];
  session_id: string;
  message_id: string;
  pilot_mode: boolean;
  original_query: string;
  matched_source: string;
};

/* ------------------------------------------------------------------ */
/*  Source label config (auto-routing用)                                */
/* ------------------------------------------------------------------ */

const SOURCE_LABELS: Record<string, { label: string; emoji: string }> = {
  faq: { label: "よくある質問", emoji: "💬" },
  spiritual: { label: "スピリチュアル相談", emoji: "✨" },
  salon: { label: "サロンFAQ", emoji: "🏠" },
};

/* ------------------------------------------------------------------ */
/*  Constants                                                          */
/* ------------------------------------------------------------------ */

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

const FALLBACK_KEYWORDS = [
  "お繋ぎ",
  "おつなぎ",
  "担当者",
  "サポート担当",
  "お問い合わせ",
  "個別相談LINE",
];

function isFallbackAnswer(text: string, sources: SourceFaq[]): boolean {
  if (sources.length === 0) return true;
  return FALLBACK_KEYWORDS.some((kw) => text.includes(kw)) && text.length < 80;
}

/* ------------------------------------------------------------------ */
/*  Bot avatar SVG                                                     */
/* ------------------------------------------------------------------ */

function BotAvatar({ size = 32 }: { size?: number }) {
  return (
    <svg
      className={styles.botAvatar}
      width={size}
      height={size}
      viewBox="0 0 36 36"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <circle cx="18" cy="18" r="18" fill="url(#avatarGrad)" />
      <path
        d="M18 8l1.8 5.5H26l-4.5 3.3 1.7 5.4L18 19l-5.2 3.2 1.7-5.4L10 13.5h6.2L18 8z"
        fill="#fde68a"
        opacity="0.95"
      />
      <circle cx="18" cy="18" r="6" fill="rgba(255,255,255,0.25)" />
      <defs>
        <linearGradient
          id="avatarGrad"
          x1="0"
          y1="0"
          x2="36"
          y2="36"
          gradientUnits="userSpaceOnUse"
        >
          <stop stopColor="#7c3aed" />
          <stop offset="1" stopColor="#a855f7" />
        </linearGradient>
      </defs>
    </svg>
  );
}

/* ------------------------------------------------------------------ */
/*  Main component                                                     */
/* ------------------------------------------------------------------ */

export default function ChatWidget() {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      role: "bot",
      text: "ご質問があればお気軽にどうぞ ✨\n\n講座・サロン・スピリチュアル用語など、なんでもお聞きください。最適なカテゴリから自動でお答えします。",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [expandedSources, setExpandedSources] = useState<
    Record<string, boolean>
  >({});
  const listRef = useRef<HTMLDivElement>(null);

  /* Auto-scroll */
  useEffect(() => {
    if (listRef.current) {
      listRef.current.scrollTop = listRef.current.scrollHeight;
    }
  }, [messages, loading]);

  /* ---- API ---- */

  const callChatApi = async (
    message: string,
    detailLevel: "concise" | "detailed",
  ): Promise<ChatApiResponse> => {
    const res = await fetch(`${API_BASE_URL}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        session_id: sessionId,
        channel_type: "web",
        detail_level: detailLevel,
        source: "auto",
      }),
    });
    if (!res.ok) throw new Error(`Chat API error: ${res.status}`);
    return (await res.json()) as ChatApiResponse;
  };

  const sendUserMessage = async () => {
    const text = input.trim();
    if (!text || loading) return;
    setMessages((prev) => [
      ...prev,
      { id: `u-${Date.now()}`, role: "user", text },
    ]);
    setInput("");
    setLoading(true);
    try {
      const data = await callChatApi(text, "concise");
      setSessionId(data.session_id);
      setMessages((prev) => [
        ...prev,
        {
          id: `b-${data.message_id}`,
          role: "bot",
          text: data.answer,
          serverMessageId: data.message_id,
          sources: data.sources,
          originalQuery: data.original_query,
          isFallback: isFallbackAnswer(data.answer, data.sources),
          matchedSource: data.matched_source,
        },
      ]);
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          id: `e-${Date.now()}`,
          role: "bot",
          text: "申し訳ありません。通信に失敗しました。少し経ってからもう一度お試しください。",
          isFallback: true,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const requestMoreDetail = async (msg: Message) => {
    if (loading || !msg.originalQuery) return;
    setMessages((prev) =>
      prev.map((m) => (m.id === msg.id ? { ...m, detailRequested: true } : m)),
    );
    setMessages((prev) => [
      ...prev,
      {
        id: `u-detail-${Date.now()}`,
        role: "user",
        text: "もっと詳しく教えてください",
      },
    ]);
    setLoading(true);
    try {
      const data = await callChatApi(msg.originalQuery, "detailed");
      setSessionId(data.session_id);
      setMessages((prev) => [
        ...prev,
        {
          id: `b-${data.message_id}`,
          role: "bot",
          text: data.answer,
          serverMessageId: data.message_id,
          sources: data.sources,
          originalQuery: data.original_query,
          isFallback: isFallbackAnswer(data.answer, data.sources),
          matchedSource: data.matched_source,
        },
      ]);
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          id: `e-${Date.now()}`,
          role: "bot",
          text: "申し訳ありません。通信に失敗しました。少し経ってからもう一度お試しください。",
          isFallback: true,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const submitFeedback = async (msg: Message, rating: Rating) => {
    if (!msg.serverMessageId || !sessionId || msg.rating) return;
    setMessages((prev) =>
      prev.map((m) => (m.id === msg.id ? { ...m, rating } : m)),
    );
    try {
      await fetch(`${API_BASE_URL}/api/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          message_id: msg.serverMessageId,
          rating,
        }),
      });
    } catch {
      /* feedback failure is non-critical */
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendUserMessage();
    }
  };

  const toggleSource = (msgId: string) => {
    setExpandedSources((prev) => ({ ...prev, [msgId]: !prev[msgId] }));
  };

  /* ---- Render ---- */

  return (
    <>
      {/* Launcher FAB */}
      <button
        type="button"
        className={styles.launcher}
        onClick={() => setOpen((v) => !v)}
        aria-label={open ? "チャットを閉じる" : "チャットを開く"}
      >
        {open ? (
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
            <path
              d="M18 6L6 18M6 6l12 12"
              stroke="currentColor"
              strokeWidth="2.5"
              strokeLinecap="round"
            />
          </svg>
        ) : (
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none">
            <path
              d="M21 11.5a8.38 8.38 0 01-.9 3.8 8.5 8.5 0 01-7.6 4.7 8.38 8.38 0 01-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 01-.9-3.8 8.5 8.5 0 014.7-7.6 8.38 8.38 0 013.8-.9h.5a8.48 8.48 0 018 8v.5z"
              fill="currentColor"
            />
          </svg>
        )}
      </button>

      {/* Widget panel */}
      {open && (
        <div
          className={styles.widget}
          role="dialog"
          aria-label="サポートチーム"
        >
          {/* Header */}
          <header className={styles.header}>
            <div className={styles.headerInner}>
              <BotAvatar size={30} />
              <span className={styles.headerTitle}>サポートチーム</span>
            </div>
            <button
              type="button"
              className={styles.headerClose}
              onClick={() => setOpen(false)}
              aria-label="閉じる"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
                <path
                  d="M18 6L6 18M6 6l12 12"
                  stroke="currentColor"
                  strokeWidth="2.5"
                  strokeLinecap="round"
                />
              </svg>
            </button>
          </header>

          {/* Messages */}
          <div className={styles.messages} ref={listRef}>
            {messages.map((msg) => {
              if (msg.role === "user") {
                return (
                  <div key={msg.id} className={styles.rowUser}>
                    <div className={styles.bubbleUser}>{msg.text}</div>
                  </div>
                );
              }
              /* Bot message */
              return (
                <div key={msg.id} className={styles.rowBot}>
                  <div className={styles.botHeader}>
                    <BotAvatar size={28} />
                    <span className={styles.botLabel}>サポートチーム</span>
                  </div>
                  <div className={styles.bubbleBot}>{msg.text}</div>

                  {/* Matched source tag */}
                  {msg.matchedSource && SOURCE_LABELS[msg.matchedSource] && (
                    <div className={styles.matchedSourceTag}>
                      <span className={styles.sourceEmoji}>
                        {SOURCE_LABELS[msg.matchedSource].emoji}
                      </span>
                      {SOURCE_LABELS[msg.matchedSource].label}
                    </div>
                  )}

                  {/* Source accordion */}
                  {msg.sources && msg.sources.length > 0 && (
                    <div className={styles.sources}>
                      <button
                        type="button"
                        className={styles.sourcesToggle}
                        onClick={() => toggleSource(msg.id)}
                      >
                        {expandedSources[msg.id] ? "▼" : "▶"} 参考FAQ (
                        {msg.sources.length}件)
                      </button>
                      {expandedSources[msg.id] && (
                        <ul className={styles.sourceList}>
                          {msg.sources.map((s, i) => (
                            <li key={i} className={styles.sourceItem}>
                              <div className={styles.sourceCategory}>
                                {s.category}
                              </div>
                              <div className={styles.sourceQuestion}>
                                Q. {s.question}
                              </div>
                            </li>
                          ))}
                        </ul>
                      )}
                    </div>
                  )}

                  {/* Action row */}
                  {!msg.id.startsWith("welcome") && (
                    <div className={styles.actionRow}>
                      {msg.originalQuery &&
                        !msg.detailRequested &&
                        !msg.isFallback && (
                          <button
                            type="button"
                            className={styles.detailButton}
                            onClick={() => requestMoreDetail(msg)}
                            disabled={loading}
                          >
                            📖 もっと詳しく
                          </button>
                        )}
                      {msg.serverMessageId && (
                        <div className={styles.feedback}>
                          <button
                            type="button"
                            className={`${styles.feedbackButton} ${
                              msg.rating === "up" ? styles.feedbackActive : ""
                            }`}
                            onClick={() => submitFeedback(msg, "up")}
                            disabled={!!msg.rating}
                            aria-label="役に立った"
                          >
                            👍
                          </button>
                          <button
                            type="button"
                            className={`${styles.feedbackButton} ${
                              msg.rating === "down" ? styles.feedbackActive : ""
                            }`}
                            onClick={() => submitFeedback(msg, "down")}
                            disabled={!!msg.rating}
                            aria-label="役に立たなかった"
                          >
                            👎
                          </button>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              );
            })}

            {/* Typing indicator */}
            {loading && (
              <div className={styles.rowBot}>
                <div className={styles.botHeader}>
                  <BotAvatar size={28} />
                  <span className={styles.botLabel}>サポートチーム</span>
                </div>
                <div className={styles.typingBubble}>
                  <span className={styles.typingDots}>
                    <span></span>
                    <span></span>
                    <span></span>
                  </span>
                  <span className={styles.typingText}>回答を準備中...</span>
                </div>
              </div>
            )}
          </div>

          {/* Input area */}
          <div className={styles.inputArea}>
            <textarea
              className={styles.textarea}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="ご質問を入力..."
              rows={1}
              disabled={loading}
            />
            <button
              type="button"
              className={styles.sendButton}
              onClick={sendUserMessage}
              disabled={loading || input.trim().length === 0}
              aria-label="送信"
            >
              <svg width="18" height="18" viewBox="0 0 20 20" fill="none">
                <path d="M3 10l14-7-4 7 4 7L3 10z" fill="currentColor" />
              </svg>
            </button>
          </div>
        </div>
      )}
    </>
  );
}
