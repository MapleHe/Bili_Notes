1. make sure the download text and show text will work with all finished text, ignore unfinished ones, but let users know that there are unfinished ones.
2. The summarization doesn't work, check whether it's because I didn't provide prompt, or because there is any bug. It should be able to summarize without user-provided prompt.
3. The configuration should be in a separate page. with the "save" button
4. In the main page, there will only be BV id input text frame. the process button.
5. Check whether the out files exist first for a specific BV ID, before any processing like downloading, or format converting, or transcription or llm processing. If transcript and summary text exist, don't download. If only transcript exists, only do summarization.
6. The ffmpeg, speech-to-text transcription should be independent with audio download. They can be run in parallel, but be aware of the memory and cpu usage. This is on a mobile with low CPU and memory.
7. Currently the model is no efficient, change the model if you find another latest updated and efficient and accurate one for Chinese.
8. Load the model when the web server start, release it (from memory) when the server stop. Think carefully before designing and develop the model preloading function, avoid memory-leak. Note that the server might be ended with "kill" command or `ctrl+c` .
