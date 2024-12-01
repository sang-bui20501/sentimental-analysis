import whisperx
import jiwer
import os

# **Configuration**
audio_file = "./dataset/test-case-2.mp3"  # Replace with your MP3 file
reference_text = """Vì sao bạn không nên chơi chứng khoán? Nếu mà bạn càng trẻ thì bạn càng không nên đầu tư chứng khoán. Mình biết là hiện giờ chứng khoán đang là trend. Ai mà không biết đầu tư chứng khoán thì ngão là quê mùa. Nhưng mà để mình chia sẻ 3 lý do vì sao mà mình nghĩ các bạn không nên chơi chứng khoán. Thứ nhất, chứng khoán là zero-sum game. Tức là có người thắng sẽ có người thua. Đừng nghĩ là tại vì các bạn đọc được vài, ba cuốn sách nên các bạn nghĩ là các bạn giỏi hơn thị trường để mà thắng được cái game này. Nhưng mà mình giả sử cho các bạn là các bạn rất giỏi luôn để các bạn đánh là thắng, đánh là thắng. Thì lý do thứ 2 là phần lớn các bạn sẽ không có đủ vốn để đầu tư để khi mà các bạn thắng, cái số tiền lời mà các bạn có được nó có ý nghĩa. Mình giả sử các bạn có 50 triệu để đầu tư đi, các bạn đầu tư giỏi lắm, giỏi hơn Warren Buffett luôn, lợi được 20% một năm. Bạn lợi được 10 triệu, 10 triệu nó có thay đổi cuộc đời các bạn không? Không, không một chút. Và lý do thứ 3, thì hôm bữa mình kêu là thay vì các bạn đầu tư vào chính khoán, các bạn hãy đầu tư vào bản thân các bạn. bỏ tiền ra để mà học thêm kỹ năng, học thêm kiếm thức các bạn nó sẽ lời nhiều hơn thì nhiều bạn không đồng ý, kêu là đầu tư chứng khoán mới đúng đầu tư từ từ, từ sớm nó sẽ chắc chắn là có lời không sai, các bạn đầu tư chứng khoán từ sớm, đầu tư từ từ chắc chắn là các bạn sẽ trở thành triệu phú lúc mà các bạn già, lúc mà các bạn 60 tuổi'"""  # Replace with the actual text
models_to_test = ["large-v2","tiny", "base", "small", "medium"]  # List of models to test
# **Functions for Evaluation**
def load_model_and_transcribe(model_name, audio_file, language="vi", compute_type="float32", device="cpu"):
    # print(f"Testing model: {model_name} with Language: {language}")
    model = whisperx.load_model(model_name, device, compute_type=compute_type)
    
    audio = whisperx.load_audio(audio_file)
    try:
        transcription_result = model.transcribe(audio, batch_size=8, language=language)
        return "".join(segment["text"] for segment in transcription_result.get("segments", [])), transcription_result.get("segments", [])
    except Exception as e:
        print(f"Error transcribing with model {model_name}: {e}")
        return "", []

def compute_accuracy(reference, hypothesis):
    """Computes Word Error Rate (WER) and Sentence Error Rate (SER)."""
    wer = jiwer.wer(reference, hypothesis)
    mer = jiwer.mer(reference, hypothesis)
    wil = jiwer.wil(reference, hypothesis)
    return {"WER": wer, "MER": mer, "WIL": wil}

def analyze_sentence_breaks(hypothesis, segments):
    """Analyzes sentence-breaking performance."""
    num_sentences = len(hypothesis.split("."))
    segment_count = len(segments)
    return {"NumSentences": num_sentences, "NumSegments": segment_count}

# **Evaluation Loop**
results = []
device = "cpu"

import time

for model_name in models_to_test:
    # Start the timer
    start_time = time.time()
    
    # Transcribe the audio and get hypothesis and segments
    hypothesis, segments = load_model_and_transcribe(
        model_name, audio_file, language="vi", compute_type="float32", device=device
    )
    
    # Stop the timer
    end_time = time.time()
    execution_time = end_time - start_time  # Calculate elapsed time

    # Check if transcription was successful
    if not hypothesis:
        print(f"Model {model_name} produced no transcription. Execution time: {execution_time:.2f} seconds.")
        continue

    # Compute accuracy metrics
    accuracy_metrics = compute_accuracy(reference_text, hypothesis)

    # Analyze sentence breaks
    sentence_analysis = analyze_sentence_breaks(hypothesis, segments)

    # Save results
    results.append({
        "Model": model_name,
        "Hypothesis": hypothesis,
        "Segments": segments,
        "Accuracy": accuracy_metrics,
        "SentenceAnalysis": sentence_analysis,
        "ExecutionTime": execution_time,  # Add execution time to results
    })
# Display results
for result in results:
    print("\n============================")
    print(f"Model: {result['Model']}")
    print(f"Transcription: {result['Hypothesis']}")
    print(f"Accuracy: {result['Accuracy']}")
    print(f"Sentence Analysis: {result['SentenceAnalysis']}")
    print(f"Execution Time: {result['ExecutionTime']:.2f} seconds")
    print(f"Segments: {result['Segments']}")