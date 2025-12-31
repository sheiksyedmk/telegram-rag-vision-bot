import logging
import asyncio
import config

from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from telegram import Update
from rag import retrieve_cached, init_db, index_docs
from prompts import build_prompt, ask_llm
from vision import describe_image  # NEW: Image processing

# Simple in-memory history
user_history = {}
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Greets the user
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    await update.message.reply_text(
        "Hi! I am a Mini-RAG bot.\nUse /ask <your question> to query the knowledge base."
    )

# /help command
async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command"""
    await update.message.reply_text(
        """/ask <question> — Ask questions from the knowledge base.
/summarize — Summarize your chats.
/history — List your last 3 interactions.
/image — Process image for description and tags."""
    )

 # /ask command
async def ask(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /ask command"""
    user_id = update.effective_user.id
    
    if not context.args:
        await update.message.reply_text("Usage: /ask <your question>")
        return

    query = " ".join(context.args)
    

    try:
        before_hits = retrieve_cached.cache_info().hits
        chunks = retrieve_cached(query)
        after_hits = retrieve_cached.cache_info().hits
        from_cache = after_hits > before_hits
        if not chunks:
            await update.message.reply_text(
                "No relevant information found in the documents."
            )
            return
        if from_cache:
            await update.message.reply_text("Answering from memory...")
        else:
            await update.message.reply_text("Thinking...")
        context_text = "\n\n".join(chunks)
        prompt = build_prompt(context_text, query)
        answer = await asyncio.to_thread(ask_llm, prompt)

        # Save to user history (keep last 3)
        if user_id not in user_history:
            user_history[user_id] = []
        user_history[user_id].append({"q": query, "a": answer})
        user_history[user_id] = user_history[user_id][-3:]

        sources = "\n".join([f"- {c[:80]}..." for c in chunks])
        await update.message.reply_text(f"{answer}\n\nSources:\n{sources}")
    except Exception as e:
        logger.exception("Error in /ask")
        await update.message.reply_text(f"Error while answering: {e}")

# summarize
async def summarize(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /summarize command"""
    user_id = update.effective_user.id
    history = user_history.get(user_id, [])
    
    if not history:
        await update.message.reply_text("No conversation history to summarize.")
        return
    
    # Build conversation text
    conv_text = ""
    for i, h in enumerate(history, 1):
        conv_text += f"Q{i}: {h['q']}\nA{i}: {h['a']}\n\n"
    
    prompt = f"""Summarize this conversation in 2-3 sentences:

{conv_text}

Summary:"""
    
    await update.message.reply_text("Summarizing...")
    summary = await asyncio.to_thread(ask_llm, prompt)
    await update.message.reply_text(f"**Conversation Summary:**\n{summary}")

# /history command
async def history_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /history command"""
    user_id = update.effective_user.id
    history = user_history.get(user_id, [])
    
    if not history:
        await update.message.reply_text("No conversation history yet. Try /ask first!")
        return
    
    response = f"**Your recent {len(history)} interactions:**\n\n"
    for i, h in enumerate(history, 1):
        response += f"{i}. Q: {h['q'][:60]}...\n   A: {h['a'][:80]}...\n\n"
    
    await update.message.reply_text(response)

# /image command
async def image_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /image command"""
    await update.message.reply_text(
        "Please send me a photo! I'll describe it and give tags.\n"
        "Or use /ask for text questions."
    )

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Process uploaded images"""
    user_id = update.effective_user.id
    
    if not update.message.photo:
        return
        
    # Get largest photo version
    photo = update.message.photo[-1]
    await update.message.reply_text("Analyzing image...")
    
    try:
        # Download image
        photo_file = await context.bot.get_file(photo.file_id)
        image_bytes = await photo_file.download_as_bytearray()
        
        # Describe with vision model
        result = describe_image(image_bytes)
        
        # Save to history
        if user_id not in user_history:
            user_history[user_id] = []
        user_history[user_id].append({
            "q": "Image analysis", 
            "a": f"Caption: {result['caption']}\nTags: {', '.join(result['tags'])}"
        })
        user_history[user_id] = user_history[user_id][-3:]
        
        # Format response
        response = f"**📸 Image Description:**\n{result['caption']}\n\n"
        if result['tags']:
            response += f"**🏷️ Tags:** {', '.join(result['tags'])}\n\n"
        response += "💬 Use /ask for text questions or /history to review."
        
        await update.message.reply_text(response)
        logger.info(f"User {user_id} image analyzed: {len(result['tags'])} tags")
        
    except Exception as e:
        logger.exception("Image processing error")
        await update.message.reply_text(f"Error analyzing image: {e}")

def main():
    if not config.TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set in config.py")

    init_db()
    index_docs()

    app = Application.builder().token(config.TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("ask", ask))
    app.add_handler(CommandHandler("summarize", summarize))
    app.add_handler(CommandHandler("history", history_cmd))
    app.add_handler(CommandHandler("image", image_cmd))  # NEW
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    logger.info("Bot starting...")
    app.run_polling()


if __name__ == "__main__":
    main()
