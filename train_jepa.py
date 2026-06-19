from config.configJEPA import ConfigJEPA
from trainer.trainerJEPA import TrainerJEPA

if __name__ == '__main__':
    cfg     = ConfigJEPA('train')
    trainer = TrainerJEPA(cfg)
    trainer.train()