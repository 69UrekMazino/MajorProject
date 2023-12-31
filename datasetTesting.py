import cv2
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Define the dataset class to load video frames and labels
class VideoDataset(Dataset):
    def __init__(self, video_filenames, hr_directory, frame_count):
        self.video_filenames = video_filenames
        self.hr_directory = hr_directory
        self.frame_count = frame_count
        self.frames_and_labels = [50]  # List to hold (frame, label) tuples

        for video_file in self.video_filenames:
            video_path = os.path.join(video_directory, video_file)
            hr_path = os.path.join(hr_directory, f"HR_{video_file.split('.')[0]}.csv")

            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            labels_df = pd.read_csv(hr_path)  # Read CSV for the video

            for frame_index in range(frame_count):
                ret, frame = cap.read()
                if not ret:
                    break

                # Extract corresponding label from the CSV file
                label = labels_df.iloc[frame_index]  # Adjust as per your CSV structure

                self.frames_and_labels.append((frame, label))

            cap.release()  # Release video capture resource after processing

        # Save frames and labels to a CSV file
        self.save_to_csv()
        # Inside the __init__ method of VideoDataset class
        print(f"Length of frames: {len(self.frames_and_labels)}")
        print(f"Length of labels: {len(labels_df)}")


    def save_to_csv(self):
        # Convert frames and labels to a DataFrame
        frames = [pair[0] for pair in self.frames_and_labels]
        labels = [pair[1] for pair in self.frames_and_labels]
        frames_df = pd.DataFrame(frames)  # Convert frames to DataFrame
        labels_df = pd.DataFrame(labels)  # Convert labels to DataFrame

        # Combine frames and labels
        combined_df = pd.concat([frames_df, labels_df], axis=1)

        # Save the combined DataFrame to a CSV file
        combined_df.to_csv('combined_data.csv', index=False)

    def __len__(self):
        return len(self.frames_and_labels)

    def __getitem__(self, idx):
        frame, label = self.frames_and_labels[idx]
        # Convert frames and labels to torch tensors if needed
        # ...

        return frame, label

video_directory = 'C:/Users/Acer/Desktop/MD/Dataset/videos/'
hr_directory = 'C:/Users/Acer/Desktop/MD/Dataset/HR/'

train_video_filenames = ['video1.MOV', 'video2.MOV']
val_video_filenames = ['video3.MOV', 'video4.MOV']
frame_count = 50  
train_dataset = VideoDataset(train_video_filenames, hr_directory, frame_count)
val_dataset = VideoDataset(val_video_filenames, hr_directory, frame_count)
batch_size = 10
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define LSTM model architecture
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) 
        return out
input_size = 3  
hidden_size = 64
num_layers = 2
learning_rate = 0.001
num_epochs = 10

# Initialize the LSTM model
model = LSTMModel(input_size, hidden_size, num_layers)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (videos, labels) in enumerate(train_loader):
        # Move data to GPU if available
        videos, labels = videos.to(device), labels.to(device)

        # Forward pass
        outputs = model(videos.float())

        # Calculate loss
        loss = criterion(outputs.squeeze(), labels.float())  # Assuming labels are scalar

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluation on validation data
    model.eval()
    with torch.no_grad():
        for val_videos, val_labels in val_loader:
            val_videos, val_labels = val_videos.to(device), val_labels.to(device)

            val_outputs = model(val_videos.float())
            val_loss = criterion(val_outputs.squeeze(), val_labels.float())
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}, Val Loss: {val_loss.item()}")

# Save the trained model
torch.save(model.state_dict(), 'heart_rate_lstm_model.pth')
