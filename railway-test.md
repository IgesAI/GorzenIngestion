# Railway Deployment Test Guide

## Quick Test Process

1. **Check Deployment Status**
   - Your Railway app should show "ACTIVE" status
   - URL: https://gorzenIngestion-production.up.railway.app

2. **Environment Variables Check**
   In Railway dashboard → Variables tab, ensure you have:
   ```
   PINECONE_API_KEY=your_actual_key_here
   OPENAI_API_KEY=your_actual_key_here (optional)
   PINECONE_CLOUD=aws
   PINECONE_REGION=us-east-1
   NODE_ENV=production
   PORT=3000
   ```

3. **Test the Application**
   - Visit your Railway URL
   - Upload a small PDF file (1-2 pages)
   - Use index name: `test-docs-$(date)`
   - Enter your Pinecone API key
   - Start processing

## Common Error Solutions

### "Exit code 127" 
- **Cause**: Python not found
- **Fix**: Railway deployment files (Dockerfile) handle this - redeploy if needed

### "PINECONE_API_KEY not set"
- **Cause**: Missing environment variable
- **Fix**: Add PINECONE_API_KEY in Railway Variables tab

### "Failed to create index"
- **Cause**: API key invalid or insufficient permissions
- **Fix**: Verify API key in Pinecone console

### "Index name validation failed"
- **Cause**: Invalid characters in index name
- **Fix**: Use only lowercase letters, numbers, hyphens (e.g., `my-docs-2024`)

## Debugging Steps

1. **Check Railway Logs**:
   - Railway Dashboard → your project → "View Logs"
   - Look for Python/Pinecone errors

2. **Verify API Keys**:
   - Test Pinecone key at: https://app.pinecone.io/
   - Test OpenAI key at: https://platform.openai.com/

3. **Try Smaller Files First**:
   - Start with 1-2 page PDFs
   - Avoid large files (>10MB) initially

## Success Indicators

✅ Railway shows "Deployment successful"  
✅ App loads at your Railway URL  
✅ File upload area appears  
✅ Can enter API keys without errors  
✅ Processing starts and shows progress  
✅ Completes with "Vector Database Created Successfully!"
