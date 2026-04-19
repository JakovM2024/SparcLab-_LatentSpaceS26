import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from latent_encoder import Encoder, Decoder

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
BRIGHT_PENALTY = 20  # how many times harder to miss a bright pixel than a dark one


class Trainer():
    def __init__(self, safe_path: str, unsafe_path: str, epochs: int = 5):
        safe_data = np.load(safe_path)
        unsafe_data = np.load(unsafe_path)

        # combine safe and unsafe images into one dataset
        all_states = np.concatenate([safe_data["states"], unsafe_data["states"]], axis=0)

        # (N, 84, 84, 3) uint8 tensor
        self.dataset = TensorDataset(torch.from_numpy(all_states))
        self.epochs = epochs

    def train(self, encoder: Encoder, decoder: Decoder):
        loader = DataLoader(self.dataset, batch_size=BATCH_SIZE, shuffle=True)
        optimizer = optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()),
            lr=LEARNING_RATE
        )
        for epoch in range(self.epochs):
            total_loss = 0.0

            for (batch,) in loader:
                z = encoder(batch)
                reconstructed = decoder(z)

                target = batch.float() / 255.0

                # weight errors by target brightness: missing a bright pixel
                # is penalized BRIGHT_PENALTY times more than a false bright on black
                weight = 1.0 + BRIGHT_PENALTY * target
                loss = (weight * (reconstructed - target).pow(2)).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch + 1}/{self.epochs}  loss: {avg_loss:.4f}")


if __name__ == "__main__":
    encoder = Encoder(latent_size=16)
    decoder = Decoder(latent_size=16)

    trainer = Trainer(
        "data/trajectories/safe_policy.npz",
        "data/trajectories/unsafe_policy.npz",
        epochs=15
    )
    trainer.train(encoder, decoder)

    torch.save(encoder.state_dict(), "data/models/encoder.pt")
    torch.save(decoder.state_dict(), "data/models/decoder.pt")
    print("Models saved.")
