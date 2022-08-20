UI_FILE = 'data/window.ui'

IMAGES = {
    'dir': 'data/images/',
    'files': {
        'Eren Yeager': 'Eren.jpg',
        'Mikasa Ackerman': 'Mikasa.jpg',
        'Anime Boy': 'Animeboy.jpg',
        'Anime Girl': 'Animegirl.jpg',
        'Okabe Rintaro': 'okabe.jpg'
    }
}

get_file_path = lambda name: IMAGES['dir'] + IMAGES['files'][name]

KERNEL_SIZE = [
    ('3 x 3', '3x3'),
    ('5 x 5', '5x5')
]


SMOOTH_LINEAR_FILTERS = [
    ('Box Filter', 'box'),
    ('Average Filter', 'avg')
]

SMOOTH_NON_LINEAR_FILTERS = [
    ('Min Filter', 'min'),
    ('Max Filter', 'max'),
    ('Mean Filter', 'mean')
]

EDGE_DETECTION_FILTERS = [
    ('Prewitt Filter', 'prewitt'),
    ('Sobel Filter', 'sobel'),
    ('Canny Filter', 'canny')
]

EDGE_DETECTION_TYPES = [
    ('Vertical', 'ver'),
    ('Horizontal', 'hor'),
    ('Both', 'both')
]

TRANS_TIME_CHANNELS = [
    ('grayscale', 1),
    ('RGB', 3),
    ('RGBA', 4)
]

