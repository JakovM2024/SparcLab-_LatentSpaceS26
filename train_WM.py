import argparse
import copy
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from latent_encoder import Encoder, Decoder
from data_utils import load_split

DEVICE = torch.device("cude" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
BRIGHT_PENALTY = 20  # how many times harder to miss a bright pixel than a dark one


def weighted_mse(reconstructed, batch):
    """Weight errors by target brightness: missing a bright pixel is penalized
    BRIGHT_PENALTY times more than a false bright on black."""
    target = batch.float() / 255.0
    weight = 1.0 + BRIGHT_PENALTY * target
    return (weight * (reconstructed - target).pow(2)).mean()


class Trainer():
    def __init__(self, train_obs: np.ndarray, val_obs: np.ndarray, epochs: int = 5):
        # (N, 84, 84, 3) uint8 tensors
        self.train_set = TensorDataset(torch.from_numpy(train_obs))
        self.val_set = TensorDataset(torch.from_numpy(val_obs)) if len(val_obs) else None
        self.epochs = epochs

    @torch.no_grad()
    def evaluate(self, encoder: Encoder, decoder: Decoder):
        if self.val_set is None:
            return None
        encoder.eval(); decoder.eval()
        total, n = 0.0, 0
        for (batch,) in DataLoader(self.val_set, batch_size=256):
            batch = batch.to(DEVICE)
            total += weighted_mse(decoder(encoder(batch)), batch).item() * len(batch)
            n += len(batch)
        encoder.train(); decoder.train()
        return total / n

    def train(self, encoder: Encoder, decoder: Decoder):
        """Returns the encoder/decoder state dicts with the lowest validation loss
        (or the final ones if there is no validation set)."""
        loader = DataLoader(self.train_set, batch_size=BATCH_SIZE, shuffle=True)
        optimizer = optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()),
            lr=LEARNING_RATE
        )
        best_val, best = float("inf"), None

        for epoch in range(self.epochs):
            total_loss = 0.0
            for (batch,) in loader:
                loss = weighted_mse(decoder(encoder(batch)), batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            train_loss = total_loss / len(loader)
            val_loss = self.evaluate(encoder, decoder)
            msg = f"Epoch {epoch + 1}/{self.epochs}  train loss: {train_loss:.4f}"
            if val_loss is not None:
                msg += f"  val loss: {val_loss:.4f}"
                if val_loss < best_val:
                    best_val = val_loss
                    best = (copy.deepcopy(encoder.state_dict()), copy.deepcopy(decoder.state_dict()))
                    msg += "  (best)"
            print(msg)

        if best is None:
            best = (encoder.state_dict(), decoder.state_dict())
        return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the image encoder/decoder.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--val-fraction", type=float, default=0.1,
                        help="fraction of episodes per dataset held out for validation (0 disables)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    train, val = load_split(val_fraction=args.val_fraction, seed=args.seed, keys=("obs",))
    print(f"{len(train['obs'])} train frames ({len(train['episode_starts']) - 1} episodes), "
          f"{len(val['obs'])} val frames ({len(val['episode_starts']) - 1} episodes)")

    encoder = Encoder(latent_size=16).to(DEVICE)
    decoder = Decoder(latent_size=16).to(DEVICE)
    trainer = Trainer(train["obs"], val["obs"], epochs=args.epochs)
    enc_state, dec_state = trainer.train(encoder, decoder)

    torch.save(enc_state, "data/models/encoder.pt")
    torch.save(dec_state, "data/models/decoder.pt")
    print("Models saved (best validation epoch)." if len(val["obs"]) else "Models saved.")
