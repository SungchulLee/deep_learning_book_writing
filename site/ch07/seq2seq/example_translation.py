"""
Complete Example: English to French Translation
Demonstrates the full pipeline from data preparation to inference
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from encoder import BasicEncoder
from decoder import AttentionDecoder
from seq2seq_model import Seq2SeqAttention
from data_preprocessing import Tokenizer, Vocabulary, ParallelDataset, split_data
from train import Seq2SeqTrainer, collate_fn, initialize_weights, count_parameters
from inference import Seq2SeqInference, BLEU


def main():
    """Main training and evaluation pipeline"""
    
    print("=" * 60)
    print("Seq2Seq Translation Example: English to French")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # ========== Step 1: Prepare Data ==========
    print("\n" + "=" * 60)
    print("Step 1: Preparing Data")
    print("=" * 60)
    
    # Sample English-French parallel corpus
    en_texts = [
        "Hello, how are you?",
        "I am fine, thank you.",
        "What is your name?",
        "My name is John.",
        "Nice to meet you.",
        "Where are you from?",
        "I am from Paris.",
        "Do you speak English?",
        "Yes, I speak English.",
        "What time is it?",
        "It is three o'clock.",
        "I like to read books.",
        "She is a teacher.",
        "He works in a hospital.",
        "We are students.",
        "They are learning French.",
        "I have a cat.",
        "The weather is nice today.",
        "Can you help me?",
        "Of course, I can help you.",
        "Thank you very much.",
        "You are welcome.",
        "Good morning.",
        "Good night.",
        "See you later.",
        "How old are you?",
        "I am twenty years old.",
        "What do you do?",
        "I am a doctor.",
        "Where do you live?",
    ]
    
    fr_texts = [
        "Bonjour, comment allez-vous?",
        "Je vais bien, merci.",
        "Quel est votre nom?",
        "Je m'appelle John.",
        "Enchanté.",
        "D'où venez-vous?",
        "Je viens de Paris.",
        "Parlez-vous anglais?",
        "Oui, je parle anglais.",
        "Quelle heure est-il?",
        "Il est trois heures.",
        "J'aime lire des livres.",
        "Elle est enseignante.",
        "Il travaille dans un hôpital.",
        "Nous sommes étudiants.",
        "Ils apprennent le français.",
        "J'ai un chat.",
        "Il fait beau aujourd'hui.",
        "Pouvez-vous m'aider?",
        "Bien sûr, je peux vous aider.",
        "Merci beaucoup.",
        "De rien.",
        "Bonjour.",
        "Bonne nuit.",
        "À plus tard.",
        "Quel âge avez-vous?",
        "J'ai vingt ans.",
        "Que faites-vous?",
        "Je suis médecin.",
        "Où habitez-vous?",
    ]
    
    print(f"Total samples: {len(en_texts)}")
    
    # Split data
    (train_en, train_fr), (val_en, val_fr), (test_en, test_fr) = split_data(
        (en_texts, fr_texts), train_ratio=0.7, val_ratio=0.15
    )
    
    print(f"Training samples: {len(train_en)}")
    print(f"Validation samples: {len(val_en)}")
    print(f"Test samples: {len(test_en)}")
    
    # ========== Step 2: Build Vocabularies ==========
    print("\n" + "=" * 60)
    print("Step 2: Building Vocabularies")
    print("=" * 60)
    
    # Create tokenizer
    tokenizer = Tokenizer(lower=True, remove_punct=False)
    
    # Build vocabularies
    en_vocab = Vocabulary(max_size=1000, min_freq=1)
    en_vocab.build_vocab(train_en, tokenizer.tokenize)
    
    fr_vocab = Vocabulary(max_size=1000, min_freq=1)
    fr_vocab.build_vocab(train_fr, tokenizer.tokenize)
    
    print(f"English vocabulary size: {len(en_vocab)}")
    print(f"French vocabulary size: {len(fr_vocab)}")
    
    # ========== Step 3: Create Datasets and DataLoaders ==========
    print("\n" + "=" * 60)
    print("Step 3: Creating Datasets")
    print("=" * 60)
    
    train_dataset = ParallelDataset(
        train_en, train_fr, en_vocab, fr_vocab,
        src_tokenizer=tokenizer, trg_tokenizer=tokenizer
    )
    
    val_dataset = ParallelDataset(
        val_en, val_fr, en_vocab, fr_vocab,
        src_tokenizer=tokenizer, trg_tokenizer=tokenizer
    )
    
    test_dataset = ParallelDataset(
        test_en, test_fr, en_vocab, fr_vocab,
        src_tokenizer=tokenizer, trg_tokenizer=tokenizer
    )
    
    BATCH_SIZE = 4
    PAD_IDX = en_vocab.pad_idx
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, PAD_IDX)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, PAD_IDX)
    )
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # ========== Step 4: Create Model ==========
    print("\n" + "=" * 60)
    print("Step 4: Creating Model")
    print("=" * 60)
    
    # Model hyperparameters
    ENC_EMB_DIM = 128
    DEC_EMB_DIM = 128
    HID_DIM = 256
    N_LAYERS = 2
    ENC_DROPOUT = 0.3
    DEC_DROPOUT = 0.3
    
    # Create encoder
    encoder = BasicEncoder(
        input_size=len(en_vocab),
        embedding_dim=ENC_EMB_DIM,
        hidden_size=HID_DIM,
        num_layers=N_LAYERS,
        dropout=ENC_DROPOUT,
        bidirectional=True,
        rnn_type='LSTM'
    )
    
    # Create decoder
    decoder = AttentionDecoder(
        output_size=len(fr_vocab),
        embedding_dim=DEC_EMB_DIM,
        hidden_size=HID_DIM * 2,  # *2 because encoder is bidirectional
        encoder_hidden_size=HID_DIM * 2,
        num_layers=N_LAYERS,
        dropout=DEC_DROPOUT,
        rnn_type='LSTM'
    )
    
    # Create seq2seq model
    model = Seq2SeqAttention(encoder, decoder, device, PAD_IDX).to(device)
    
    # Initialize weights
    initialize_weights(model)
    
    print(f"Model created with {count_parameters(model):,} trainable parameters")
    
    # ========== Step 5: Train Model ==========
    print("\n" + "=" * 60)
    print("Step 5: Training Model")
    print("=" * 60)
    
    # Training configuration
    LEARNING_RATE = 0.001
    N_EPOCHS = 20
    CLIP = 1.0
    
    # Create optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    
    # Create trainer
    trainer = Seq2SeqTrainer(model, optimizer, criterion, device, PAD_IDX, CLIP)
    
    # Train model
    train_losses, val_losses = trainer.train(
        train_loader,
        val_loader,
        num_epochs=N_EPOCHS,
        checkpoint_dir='checkpoints',
        teacher_forcing_ratio=0.5,
        save_every=5
    )
    
    # ========== Step 6: Evaluate Model ==========
    print("\n" + "=" * 60)
    print("Step 6: Evaluating Model")
    print("=" * 60)
    
    # Create inference object
    en_vocab_dict = en_vocab.token2idx
    fr_vocab_dict = fr_vocab.token2idx
    
    inference = Seq2SeqInference(
        model, en_vocab_dict, fr_vocab_dict, device,
        sos_idx=fr_vocab.sos_idx,
        eos_idx=fr_vocab.eos_idx,
        pad_idx=PAD_IDX
    )
    
    # Test translations
    print("\nTest Translations:")
    print("-" * 60)
    
    bleu_scores = []
    
    for i, (en_text, fr_text) in enumerate(zip(test_en, test_fr)):
        translation, _ = inference.greedy_decode(en_text, max_len=50)
        
        # Calculate BLEU score
        bleu = BLEU.compute_bleu(fr_text, translation)
        bleu_scores.append(bleu)
        
        print(f"\nExample {i+1}:")
        print(f"English:    {en_text}")
        print(f"French:     {fr_text}")
        print(f"Predicted:  {translation}")
        print(f"BLEU:       {bleu:.4f}")
    
    avg_bleu = np.mean(bleu_scores)
    print(f"\nAverage BLEU Score: {avg_bleu:.4f}")
    
    # ========== Step 7: Interactive Translation ==========
    print("\n" + "=" * 60)
    print("Step 7: Interactive Translation")
    print("=" * 60)
    
    # Example translations
    test_sentences = [
        "Hello, my friend.",
        "How are you today?",
        "I am learning French.",
        "What is the time?"
    ]
    
    print("\nAdditional Examples:")
    print("-" * 60)
    
    for sent in test_sentences:
        translation, _ = inference.greedy_decode(sent)
        print(f"EN: {sent}")
        print(f"FR: {translation}\n")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
