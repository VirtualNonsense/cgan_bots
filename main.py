import logging
import cgan_bots.cgan_bot
from secrets import bot_key
from project_cgan.lib import CDCGAN

if __name__ == '__main__':
    class_dict = {
        "bald": 0,
        "blackhair": 1,
        "brownhair": 2,
        "glasses": 3,
        "beard": 4,
    }
    filters = [1024, 512, 256, 128, 64]
    input_shape = (1, 100, 1, 1)
    image_size = 128
    color_channel = 3
    model = CDCGAN.load_from_checkpoint(
        "models/celebA_scheduler_ReduceLROnPlateau_only_g_128_100_1024-512-256-128-64-epoch=29.ckpt",
        input_dim=input_shape[1],
        amount_classes=len(class_dict),
        filter_sizes=filters,
        color_channels=color_channel,
        image_size=image_size,
        device="cpu"
    )
    model.eval()
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    b = cgan_bots.cgan_bot.CGANBot(bot_key, model, input_shape, class_dict)
    b.run()
