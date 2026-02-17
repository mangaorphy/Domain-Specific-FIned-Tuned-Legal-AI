# 🤗 HuggingFace Spaces Deployment Guide

Quick guide to deploy your Legal Case Summarization model to HuggingFace Spaces.

## 📋 Prerequisites

✅ HuggingFace account (free): https://huggingface.co/join
✅ Model files in `model_outputs/` folder
✅ Git installed locally

## 🚀 Deployment Steps

### Step 1: Create a New Space

1. Go to: https://huggingface.co/new-space
2. Fill in details:
   - **Space name**: `legal-case-summarizer` (or your choice)
   - **License**: MIT
   - **Select the Space SDK**: **Gradio** ⭐
   - **Space hardware**: Start with **CPU Basic** (free)
     - Can upgrade to GPU later if needed
   - **Visibility**: Public (recommended) or Private

3. Click **"Create Space"**

### Step 2: Clone Your HuggingFace Space

After creating the space, HuggingFace will show you a git clone command. Run it:

```bash
# Clone your space (replace with your username and space name)
git clone https://huggingface.co/spaces/YOUR_USERNAME/legal-case-summarizer
cd legal-case-summarizer
```

### Step 3: Copy Your Project Files

Copy the necessary files from your project:

```bash
# From your project directory
cd /Users/cococe/Desktop/Domain-Specific-FIned-Tuned-Legal-AI

# Copy app and requirements
cp app.py ~/legal-case-summarizer/
cp requirements.txt ~/legal-case-summarizer/

# Copy README (rename HF_README.md to README.md)
cp HF_README.md ~/legal-case-summarizer/README.md

# Copy model outputs (IMPORTANT!)
cp -r model_outputs/ ~/legal-case-summarizer/
```

### Step 4: Verify Files

Your space directory should contain:

```
legal-case-summarizer/
├── app.py                    # Gradio interface
├── requirements.txt          # Dependencies
├── README.md                 # Space description (from HF_README.md)
└── model_outputs/           # Your trained model
    ├── adapter_config.json
    ├── adapter_model.safetensors
    ├── tokenizer_config.json
    └── tokenizer.json
```

### Step 5: Push to HuggingFace

```bash
cd ~/legal-case-summarizer

# Add all files
git add .

# Commit
git commit -m "Initial deployment: Legal Case Summarization with LoRA Gemma-2B"

# Push to HuggingFace
git push
```

### Step 6: Wait for Build

- HuggingFace will automatically build your Space
- First build takes 5-10 minutes
- You can watch progress in the "Logs" tab
- When complete, your app will be live!

## 🌐 Access Your Space

Your app will be available at:
```
https://huggingface.co/spaces/YOUR_USERNAME/legal-case-summarizer
```

## ⚙️ Optional: Upgrade to GPU

If the app is too slow on CPU:

1. Go to your Space settings
2. Click **"Hardware"**
3. Select **"CPU Upgrade"** or **"T4 GPU"**
   - T4 GPU: ~$0.60/hour (pay-as-you-go)
   - Only charged when Space is running
4. Click **"Upgrade"**

GPU makes generation 5-10x faster (10 seconds vs 60 seconds).

## 🔧 Troubleshooting

### Issue: "Model files not found"

**Solution**: Make sure `model_outputs/` folder is in your space with all required files:
```bash
git lfs track "*.safetensors"
git add .gitattributes
git add model_outputs/
git commit -m "Add model files"
git push
```

### Issue: "Running out of memory"

**Solutions**:
1. Upgrade to T4 GPU (recommended)
2. Or modify `app.py` to use CPU with lower batch size
3. Or reduce `max_length` default to 128

### Issue: "Build failed"

Check the **Logs** tab for specific errors. Common fixes:
- Ensure `requirements.txt` has correct versions
- Verify all model files are present
- Check that `app.py` doesn't have local file paths

### Issue: "App is slow"

This is normal on CPU. Options:
- Upgrade to GPU hardware ($0.60/hr)
- Add progress messages so users know it's working
- Reduce default `max_length` to 128 tokens

## 📊 Monitor Your Space

HuggingFace provides:
- **Analytics**: View usage stats
- **Logs**: Debug issues in real-time
- **Community**: Users can comment and provide feedback
- **Duplicate**: Others can duplicate your Space to customize it

## 🎨 Customize Your Space

Edit directly on HuggingFace:
1. Click **"Files"** tab in your Space
2. Click any file to edit
3. Save changes
4. Space automatically rebuilds

Or edit locally and push:
```bash
# Make changes to app.py
git add app.py
git commit -m "Update interface"
git push
```

## 🔗 Share Your Space

Once live, share the URL:
- On social media
- In your GitHub README
- On your portfolio
- In academic papers

## 📈 Next Steps

1. ✅ Test your deployed Space
2. ✅ Share the URL with colleagues
3. ✅ Monitor usage and feedback
4. ✅ Consider upgrading to GPU if needed
5. ✅ Add Space to your HuggingFace profile

## 🆘 Get Help

- **HuggingFace Docs**: https://huggingface.co/docs/hub/spaces
- **Community Forum**: https://discuss.huggingface.co/
- **Discord**: https://hf.co/join/discord
- **GitHub Issues**: https://github.com/mangaorphy/Domain-Specific-FIned-Tuned-Legal-AI/issues

---

**Your Space URL will be:**
`https://huggingface.co/spaces/YOUR_USERNAME/legal-case-summarizer`

🎉 **You're ready to deploy!**
