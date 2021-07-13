import logging
import cgan_bots.cgan_bot
from secrets import bot_key
from project_cgan.lib import CDCGAN

if __name__ == '__main__':
    people_dict = {
        "Bald": 0,
        "Blackhair": 1,
        "Brownhair": 2,
        "Glasses": 3,
        "Beard": 4,
    }
    art_dict = {
        "Electronic": 0,
        "Hip-Hop": 1,
        "Jazz": 2,
        "Metal": 3,
        "Pop": 4,
    }
    filters = [1024, 512, 256, 128, 64, 32]
    input_shape = (1, 500, 1, 1)
    image_size = 256
    color_channel = 3
    people_model = CDCGAN.load_from_checkpoint(
        "models/celebA_high_res_256_500_1024-512-256-128-64-32-epoch=26.ckpt",
        input_dim=input_shape[1],
        amount_classes=len(people_dict),
        filter_sizes=filters,
        color_channels=color_channel,
        image_size=image_size,
        device="cpu"
    )
    people_model.eval()

    artwork_model = CDCGAN.load_from_checkpoint(
        "models/cover_art_high_res_dataloader_256_500_1024-512-256-128-64-32-epoch=836.ckpt",
        input_dim=input_shape[1],
        amount_classes=len(people_dict),
        filter_sizes=filters,
        color_channels=color_channel,
        image_size=image_size,
        device="cpu"
    )
    artwork_model.eval()
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    b = cgan_bots.cgan_bot.CGANBot(bot_key,
                                   people_model,
                                   input_shape,
                                   people_dict,
                                   artwork_model,
                                   input_shape,
                                   art_dict)
    b.run()
