import ChatWidget from "./components/ChatWidget";

export default function Home() {
  return (
    <main
      style={{
        minHeight: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      <div
        style={{
          textAlign: "center",
          color: "#9ca3af",
          fontSize: 15,
          lineHeight: 1.8,
          padding: 24,
        }}
      >
        <p style={{ fontSize: 40, marginBottom: 8 }}>🌐</p>
        <p>ここにWebサイトが表示されます</p>
        <p style={{ fontSize: 13 }}>
          右下のボタンからサポートチャットをお試しください
        </p>
      </div>
      <ChatWidget />
    </main>
  );
}
