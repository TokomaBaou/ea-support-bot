# FAQ Chatbot Prototype

スピリチュアル系オンライン講座の **受講生向け** FAQ チャットボット (デモ用プロトタイプ)。
受講生は年配・デジタル弱者の方が多いため、UI とプロンプトを「やさしく・簡潔に・専門用語を避ける」方向にチューニングしています。

- バックエンド: FastAPI + OpenAI API（**ファインチューニングしたGPTモデルを利用する前提**）
- フロント: Next.js 14 (App Router) + TypeScript
- FAQ ストア: JSON ファイル（`backend/data/` 配下に `faq.json` / `spiritual_faq.json` / `salon_faq.json`）
- 通知: Slack Incoming Webhook（パイロット通知 / 有人エスカレーション / 👎フィードバック）
- 併用検証中: Anthropic Claude（メソッド系・「先生らしさ」表現の精度比較用、PoCで採否を決める）

## モデル戦略（PoCの方針）

| Phase | 主モデル | ベース | 役割 |
| --- | --- | --- | --- |
| Phase 1 FAQ Bot | OpenAI ファインチューン版 | `gpt-4o-mini` | 受講生からの定型問合せ。コスト最安・低レイテンシを優先 |
| Phase 2 メソッド Bot | OpenAI ファインチューン版 | `gpt-4o` | 「藤本先生らしさ」「スピリチュアル系メソッド」の表現再現を優先 |
| 併用検証 | Anthropic Claude (Sonnet 4.6 / Haiku 4.5) | — | 同一質問でA/B比較。トーン再現精度がGPTファインチューン版を上回る場合は併用導入を検討 |

**Claude 併用の発動条件は PoC 期間中に検証して決める**（source 別 / detail_level 別 / フォールバック専用 / フィーチャーフラグでのA/B など、複数パターンを実機で比較）。本リポジトリのコードは現状 OpenAI 単体実装で、Claude併用ロジックは未実装（検証結果に応じて追加する）。

## デモのコンセプト

| 項目 | 内容 |
| --- | --- |
| ターゲット | 年配・デジタル弱者の受講生 |
| 応答方針 | 3行以内・専門用語を避けた口語・敬体 |
| エスカレーション | FAQ範囲外の質問は担当者への問い合わせを案内 |
| パイロット運用 | `PILOT_MODE=true`時は Bot 回答案を Slack に通知（担当者が手動でフォロー） |
| フィードバック | 各回答に 👍 / 👎 ボタン、👎 は Slack にも通知 |
| 「もっと詳しく」 | 同じ質問を `detail_level=detailed` で再問い合わせし、ゆっくり丁寧版を返す |
| 複数ソース対応 | 一般FAQ / スピリチュアル相談 / 入会後サロンFAQ を自動振り分け (`source=auto`) |

## ディレクトリ構成

```
faq-chatbot-prototype/
├── backend/
│   ├── main.py                # FastAPI エントリ + CORS + ルータ登録
│   ├── chat.py                # POST /api/chat (パイロットモード対応)
│   ├── escalate.py            # POST /api/escalate
│   ├── feedback.py            # POST /api/feedback
│   ├── slack.py               # Slack Incoming Webhook ヘルパー
│   ├── faq_search.py          # キーワード + 文字 bigram 簡易検索
│   ├── formatter.py           # 応答フォーマッタ (web / line)
│   ├── data/
│   │   ├── faq.json           # 一般FAQ (アカウント / 受講方法 / 決済 / 技術トラブル / 講座内容)
│   │   ├── spiritual_faq.json # スピリチュアル用語データ（相談対応の背景知識として使用）
│   │   └── salon_faq.json     # スピ覚醒サロン入会後FAQ
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── package.json
│   ├── tsconfig.json
│   ├── next.config.js
│   ├── .env.local.example
│   └── src/app/
│       ├── layout.tsx
│       ├── page.tsx
│       ├── globals.css
│       └── components/
│           ├── ChatWidget.tsx
│           └── ChatWidget.module.css
└── README.md
```

## セットアップ

### 1. バックエンド

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# .env を編集
#   OPENAI_API_KEY=sk-proj-...                              # 必須
#   CHAT_MODEL=gpt-4o-mini                                  # FAQ用、ファインチューン後はモデルIDに差し替え
#   SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...  # 任意
#   PILOT_MODE=true
#   ANTHROPIC_API_KEY=sk-ant-...                            # 任意（Claude併用検証時のみ）
```

起動:

```bash
uvicorn main:app --reload --port 8000
```

ヘルスチェック: <http://localhost:8000/health>

### 2. フロントエンド

別ターミナルで:

```bash
cd frontend
npm install
cp .env.local.example .env.local   # 必要に応じて API URL を変更
npm run dev
```

ブラウザで <http://localhost:3000> を開く。画面右下の青いボタンからチャットウィジェットが起動します。

## OpenAI ファインチューニングのワークフロー

PoC では **「藤本先生らしさ」「やさしい口調」「サロン用語の自然な使用」** を毎回プロンプトで指示する代わりに、ベースモデルにファインチューニングで焼き込む方針です。

### Phase 1 FAQ Bot 用（`gpt-4o-mini` ベース）

**学習データのソース**
- `backend/data/faq.json`（25件）
- 秘書チームの定型返信（既存20件想定）
- 運用後に蓄積されるログ（受講生の実質問 → 担当者の修正済み回答）

**JSONL 形式の例**

```jsonl
{"messages":[{"role":"system","content":"あなたはスピリチュアル系オンライン講座のサポートアシスタントです。やさしい口調で3行以内、専門用語を避けてください。"},{"role":"user","content":"パスワードを忘れてしまいました"},{"role":"assistant","content":"ログイン画面の「パスワードを忘れた方はこちら」をタップし、ご登録のメールアドレスを入力してください。再設定用のメールが届きますので、メール内のリンクから新しいパスワードをご設定くださいね。"}]}
```

**手順の概略**

```bash
# 1. データを JSONL に変換（faq.json → finetune_faq.jsonl）
python tools/build_finetune_data.py --source faq --out data/finetune_faq.jsonl   # 将来追加

# 2. OpenAI にアップロード
openai api files.create -f data/finetune_faq.jsonl -p fine-tune

# 3. ファインチューニングジョブ作成
openai api fine_tuning.jobs.create \
  -t <file_id> \
  -m gpt-4o-mini-2024-07-18 \
  --suffix enlight-academy-faq

# 4. 完了後、返ってきたモデルID（例: ft:gpt-4o-mini-2024-07-18:enlight-academy:faq:ABC123）を
#    .env の CHAT_MODEL に設定して再起動
```

### Phase 2 メソッド Bot 用（`gpt-4o` ベース）

**学習データのソース**
- 藤本先生の過去発信（投稿・LINE発信30〜50本以上）
- Vimeo講座動画の字幕テキスト
- 講座資料・著書からのQ&Aサンプル
- 「先生らしさ」のサンプル文章（5-3 で先生から提供してもらう）

**Phase 1 と分けてジョブを作成**し、`source=salon` または `source=spiritual` 時に切り替えるロジックを `chat.py` に追加する想定（現状未実装、検証段階）。

### 評価とイテレーション

- 評価セット（held-out）を 30〜50件用意し、各イテレーションで以下をスコアリング：
  - **トーン再現度**：藤本先生本人による5段階評価
  - **NG領域への侵入率**：医療・宗教系プロンプトで拒否型応答が出るか
  - **コスト**：ファインチューン版 vs ベースモデル+プロンプトのトークン量
- Claude併用が必要かどうかも、ここで決定する（同じ評価セットを Claude にもコール）

## API 仕様

### POST /api/chat

```json
// Request
{
  "message": "パスワードを忘れた",
  "session_id": "sess_abc123",
  "channel_type": "web",
  "detail_level": "concise",
  "source": "auto"
}

// Response
{
  "answer": "ログイン画面の「パスワードを忘れた方はこちら」をタップしてください。…",
  "sources": [
    {
      "question": "パスワードを忘れてしまいました。",
      "answer": "ログイン画面の…",
      "category": "アカウント"
    }
  ],
  "session_id": "sess_abc123",
  "message_id": "msg_xxxxxxxxxxxx",
  "pilot_mode": true,
  "original_query": "パスワードを忘れた",
  "matched_source": "faq"
}
```

| フィールド       | 型                                       | 必須 | 備考                                                                  |
| ---------------- | ---------------------------------------- | ---- | --------------------------------------------------------------------- |
| `message`        | string                                   | ✓    | ユーザー入力 (1〜2000 文字)                                           |
| `session_id`     | string \| null                           |      | 省略時はサーバ側で発行                                                |
| `channel_type`   | `"web"` \| `"line"`                      |      | フォーマッタ切替用 (LINE は将来追加)                                  |
| `detail_level`   | `"concise"` \| `"detailed"`              |      | 「もっと詳しく」ボタン用。デフォルト `concise` (3行以内・100字程度)   |
| `source`         | `"auto"` \| `"faq"` \| `"spiritual"` \| `"salon"` |      | 検索対象FAQセットの切替。デフォルト `auto`（全ソース横断で自動振り分け） |

### POST /api/escalate

```json
// Request
{
  "session_id": "sess_abc123",
  "question": "動画が止まってしまいます",
  "bot_answer": "ブラウザを閉じてもう一度開き直してください。",
  "student_id": "demo_student_001"
}

// Response
{
  "ok": true,
  "escalated_at": "2026-05-04T10:23:15.123456+00:00",
  "notified": true
}
```

`SLACK_WEBHOOK_URL` が設定されていれば Slack に通知。未設定時は `notified: false` を返してログに警告を出すだけで処理は成功扱い。

### POST /api/feedback

```json
// Request
{
  "session_id": "sess_abc123",
  "message_id": "msg_xxxxxxxxxxxx",
  "rating": "down",
  "comment": "回答が分かりにくかった"
}

// Response
{ "ok": true, "received_at": "2026-05-04T10:24:00+00:00" }
```

`rating: "down"` は Slack にも通知（改善対象として担当者の目に触れる）。`up` はサーバログのみ。

## 動作の流れ

1. 受講生がウィジェットに質問を入力
2. `faq_search.search_auto()` で全3ソース（faq / spiritual / salon）を横断検索し、最適なソースを自動判定（`matched_source`）
3. `matched_source` に応じたシステムプロンプトと一緒に OpenAI（ファインチューン後はファインチューン済みモデル）に投げる
4. **`PILOT_MODE=true`** の場合、Bot の回答案を Slack に通知（担当者が確認）
5. フロントは回答テキスト + ソースタグ（💬よくある質問 / ✨スピリチュアル相談 / 🏠サロンFAQ）+ 「もっと詳しく聞く」「👍/👎」ボタンを表示
6. 「もっと詳しく」を押すと、同じ質問を `detail_level=detailed` で再問い合わせ → ゆっくり丁寧版を返す

## ソース別の応答方針

| ソース | 用途 | 応答スタイル |
| --- | --- | --- |
| `faq` | 講座の受講方法・支払い・技術トラブル等 | やさしい口調で簡潔に手順を案内（FAQ回答ベース） |
| `spiritual` | スピリチュアルな体験・用語に関する相談 | 用語データを**背景知識**として活用し、体験や悩みに共感的に寄り添う相談対応モード |
| `salon` | サロン内のルール・活動・コースについて | サロンメンバーとして温かく案内（サロンFAQ回答ベース） |

### spiritual ソースの相談対応モード

`spiritual_faq.json` の用語データは「定義をそのまま返す辞書」ではなく、**相手の体験や悩みに寄り添うための背景知識**として使用します。

- 例: 「チャクラが詰まってる感じがして…」→ チャクラの知識を踏まえた上で、どう向き合えばよいかアドバイス
- 例: 「デジャヴュをよく見るんだけど」→ デジャヴュの概念を踏まえつつ、体験に共感した回答
- トーンは「〜かもしれませんね」「〜してみてはいかがでしょうか」など、共感と提案を織り交ぜた柔らかい口調

### 将来の人格再現（ファインチューニング）

現在はシステムプロンプトでトーンを制御していますが、将来的にはファインチューニングにより依頼人（藤本先生）の人格・語り口を再現する予定です。

- **Phase 1（現在）**: システムプロンプトで共感的トーンを指示
- **Phase 2（計画中）**: 藤本先生の過去発信・講座動画字幕等を学習データとしてファインチューニングし、「先生らしさ」を焼き込む
- spiritual ソースは Phase 2 で最も効果が大きい領域（メソッド系の表現・寄り添い方に先生の個性が反映される）

## UI 配慮

- 文字サイズ: ボット回答 17px、入力欄・ボタン 16px、ヘッダ 18px
- ボタン高さ: 主要ボタンは 48px 以上 (タップしやすさ重視)
- 「人に相談する」ボタンは常時オレンジ色で目立つ位置 (入力欄直上) に固定
- 専門用語をプロンプトレベルで禁止 (「クライアント」「キャッシュ」「再起動」等)
- 答えが見つからない場合は推測せず「サポート担当者にお繋ぎしますね」で締める

## 応答速度について

- **`gpt-4o-mini`（ファインチューン版含む）**: 通常応答 1〜2 秒程度。FAQ 用途では十分
- **`gpt-4o`（ファインチューン版含む）**: 通常応答 2〜4 秒程度。メソッド系の表現精度を優先
- 5 秒を超える場合は `.env` の `CHAT_MODEL` を切り替えるか、システムプロンプトを短縮する

## LINE / マルチチャネル拡張

- `channel_type` を `"line"` に切り替えると `formatter.format_for_line`（将来追加）を呼び出すだけで応答整形を差し替えられる
- 検索ロジック (`faq_search`) と LLM 呼び出し (`chat`) はチャネル非依存
- LINE Messaging API webhook を受ける薄いアダプタ層を別 Router として追加すれば、既存の `search → LLM → formatter` パイプラインをそのまま再利用できる
- UTAGE / WordPress サイト埋込時は同じバックエンドAPIを Web ウィジェットから叩く構成
- エスカレーション・フィードバックも Slack 通知ロジックを共通化済み

## Claude 併用を検証する場合（参考）

将来 Claude 併用を実装する際は、`chat.py` を以下の方向で拡張する想定です（**現状未実装**）：

1. `LLMClient` 抽象を導入し、`OpenAIClient` / `AnthropicClient` を切替可能にする
2. 環境変数や `source` 値で振り分け（例: `source=salon` → Claude Sonnet 4.6、それ以外 → OpenAI）
3. PoC ではフィーチャーフラグでセッション ID 単位の A/B を取り、トーン評価・拒否率・コストで判定

詳細は Notion ドキュメント「PoC見積もり総括 ＆ チャットbot準備」の 6-2 / 6-3 を参照。

## 試してみる質問例

- 「パスワードを忘れてしまいました」
- 「動画を再生しても画面が真っ黒で見えません」
- 「修了証はもらえますか？」
- 「領収書を発行してほしいです」
- 「課題はどうやって出せばいいですか？」
- 「猫の飼い方を教えて」（← FAQ 範囲外。担当者にお繋ぎする旨の応答を確認できます）
