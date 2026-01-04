import re

def chunk_text(text, max_length=200):
    """
    Splits long text into manageable chunks by sentences.
    
    Args:
        text (str): The input text.
        max_length (int): Maximum characters per chunk (approximate).
        
    Returns:
        list: List of text chunks.
    """
    # Split by common sentence enders and major pauses (comma, semicolon)
    sentences = re.split(r'(?<=[.!,;?])\s+', text.strip())
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_length:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            
            # If a single sentence is longer than max_length, split it by words
            if len(sentence) > max_length:
                words = sentence.split()
                sub_chunk = ""
                for word in words:
                    if len(sub_chunk) + len(word) + 1 <= max_length:
                        if sub_chunk:
                            sub_chunk += " " + word
                        else:
                            sub_chunk = word
                    else:
                        chunks.append(sub_chunk)
                        sub_chunk = word
                current_chunk = sub_chunk
            else:
                current_chunk = sentence
                
    if current_chunk:
        chunks.append(current_chunk)
        
    return chunks

if __name__ == "__main__":
    test_text = "This is a sentence. This is another sentence that is slightly longer. And here is one more sentence to test the chunking logic."
    print(chunk_text(test_text, 50))
