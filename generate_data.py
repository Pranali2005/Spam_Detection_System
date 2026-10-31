import pandas as pd
import random

# Base phrases to construct synthetic data
spam_phrases = [
    "WINNER! You have won a cash prize of $5000! Call now to claim.",
    "URGENT: Your account has been suspended. Click here to verify your details.",
    "Get rich quick! Earn $500 a day working from home. No experience needed.",
    "Congratulations! You've been selected for a free iPhone. Click the link.",
    "Dear customer, your lottery ticket has won! Reply with your bank details.",
    "Exclusive offer: Crypto secrets revealed. Multiply your investment today!"
]

ham_phrases = [
    "Hey, are we still meeting for lunch today at 1 PM?",
    "Can you send me the meeting notes when you get a chance?",
    "I'll be a bit late, stuck in traffic. Don't wait up.",
    "Thanks for the help yesterday, I really appreciate it.",
    "Don't forget to buy milk on your way back home.",
    "Are you free this weekend for a quick catch-up call?"
]

# Generate 3,000 records
data = []
for _ in range(3000):
    if random.random() > 0.7:  # ~30% Spam, 70% Ham
        text = random.choice(spam_phrases) + f" Code: {random.randint(1000, 9999)}"
        label = 1  # Spam
    else:
        text = random.choice(ham_phrases) + f" See you soon."
        label = 0  # Ham
    data.append([text, label])

# Save to CSV
df = pd.DataFrame(data, columns=['text', 'label'])
df.to_csv('spam_dataset.csv', index=False)
print("Dataset created successfully with 3000 records!")