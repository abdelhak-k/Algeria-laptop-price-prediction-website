from django import forms

# Define brand-series mapping
BRAND_SERIES_MAP = {
    'LENOVO': ['IDEAPAD', 'THINKPAD', 'LEGION', 'YOGA', 'UNKNOWN'],
    'HP': ['PAVILION', 'OMEN', 'ELITEBOOK', 'PROBOOK', 'SPECTRE', 'ENVY', 'VICTUS', 'UNKNOWN'],
    'DELL': ['INSPIRON', 'LATITUDE', 'XPS', 'ALIENWARE', 'VOSTRO', 'PRECISION', 'G SERIES', 'UNKNOWN'],
    'ASUS': ['VIVOBOOK', 'ZENBOOK', 'ROG', 'TUF', 'EXPERTBOOK', 'UNKNOWN'],
    'ACER': ['ASPIRE', 'NITRO', 'PREDATOR', 'SWIFT', 'SPIN', 'UNKNOWN'],
    'APPLE': ['MACBOOK AIR', 'MACBOOK PRO'],
    'MSI': ['STEALTH', 'KATANA', 'RAIDER', 'PRESTIGE', 'MODERN', 'UNKNOWN'],
    'GIGABYTE': ['AERO', 'AORUS', 'G5', 'UNKNOWN'],
    'HUAWEI': ['MATEBOOK', 'UNKNOWN'],
    'SAMSUNG': ['GALAXY BOOK', 'UNKNOWN'],
    'TOSHIBA': ['SATELLITE', 'TECRA', 'UNKNOWN'],
    'OTHER': ['UNKNOWN'],
}

# CPU brand to CPU family mapping
CPU_BRAND_FAMILY_MAP = {
    'INTEL': [
        ('i3', 'Intel Core i3'),
        ('i5', 'Intel Core i5'),
        ('i7', 'Intel Core i7'),
        ('i9', 'Intel Core i9'),
        ('ultra5', 'Intel Core Ultra 5'),
        ('ultra7', 'Intel Core Ultra 7'),
        ('ultra9', 'Intel Core Ultra 9'),
        ('celeron', 'Intel Celeron'),
        ('pentium', 'Intel Pentium'),
        ('xeon', 'Intel Xeon'),
    ],
    'AMD': [
        ('r3', 'AMD Ryzen 3'),
        ('r5', 'AMD Ryzen 5'),
        ('r7', 'AMD Ryzen 7'),
        ('r9', 'AMD Ryzen 9'),
    ],
    'APPLE': [
        ('m1', 'Apple M1'),
        ('m2', 'Apple M2'),
        ('m3', 'Apple M3'),
        ('m4', 'Apple M4'),
    ],
    'QUALCOMM': [
        ('SNAPDRAGON', 'Qualcomm Snapdragon'),
    ],
    'UNKNOWN': [
        ('UNKNOWN', 'Unknown'),
    ],
}

# Define choices based on the preprocessed data
BRAND_CHOICES = [
    ('', 'Select Brand'),
    ('ACER', 'Acer'),
    ('APPLE', 'Apple'),
    ('ASUS', 'Asus'),
    ('DELL', 'Dell'),
    ('GIGABYTE', 'Gigabyte'),
    ('HP', 'HP'),
    ('HUAWEI', 'Huawei'),
    ('LENOVO', 'Lenovo'),
    ('MSI', 'MSI'),
    ('SAMSUNG', 'Samsung'),
    ('TOSHIBA', 'Toshiba'),
    ('OTHER', 'Other'),
]

SERIES_CHOICES = [
    ('', 'Select Series'),
    ('IDEAPAD', 'IdeaPad'),
    ('THINKPAD', 'ThinkPad'),
    ('LEGION', 'Legion'),
    ('YOGA', 'Yoga'),
    ('PAVILION', 'Pavilion'),
    ('OMEN', 'Omen'),
    ('VICTUS', 'Victus'),
    ('ELITEBOOK', 'EliteBook'),
    ('PROBOOK', 'ProBook'),
    ('SPECTRE', 'Spectre'),
    ('ENVY', 'Envy'),
    ('INSPIRON', 'Inspiron'),
    ('LATITUDE', 'Latitude'),
    ('XPS', 'XPS'),
    ('ALIENWARE', 'Alienware'),
    ('VOSTRO', 'Vostro'),
    ('PRECISION', 'Precision'),
    ('G SERIES', 'G Series'),
    ('VIVOBOOK', 'VivoBook'),
    ('ZENBOOK', 'ZenBook'),
    ('ROG', 'ROG'),
    ('TUF', 'TUF'),
    ('EXPERTBOOK', 'ExpertBook'),
    ('ASPIRE', 'Aspire'),
    ('NITRO', 'Nitro'),
    ('PREDATOR', 'Predator'),
    ('SWIFT', 'Swift'),
    ('SPIN', 'Spin'),
    ('MACBOOK AIR', 'MacBook Air'),
    ('MACBOOK PRO', 'MacBook Pro'),
    ('AERO', 'Aero'),
    ('AORUS', 'Aorus'),
    ('G5', 'G5'),
    ('STEALTH', 'Stealth'),
    ('KATANA', 'Katana'),
    ('RAIDER', 'Raider'),
    ('PRESTIGE', 'Prestige'),
    ('MODERN', 'Modern'),
    ('MATEBOOK', 'MateBook'),
    ('GALAXY BOOK', 'Galaxy Book'),
    ('SATELLITE', 'Satellite'),
    ('TECRA', 'Tecra'),
    ('UNKNOWN', 'Unknown/Other'),
]

CPU_FAMILY_CHOICES = [
    ('', 'Select CPU Family'),
    # Intel Core
    ('i3', 'Intel Core i3'),
    ('i5', 'Intel Core i5'),
    ('i7', 'Intel Core i7'),
    ('i9', 'Intel Core i9'),
    # Intel Ultra (new)
    ('ultra5', 'Intel Core Ultra 5'),
    ('ultra7', 'Intel Core Ultra 7'),
    ('ultra9', 'Intel Core Ultra 9'),
    # Intel Other
    ('celeron', 'Intel Celeron'),
    ('pentium', 'Intel Pentium'),
    ('xeon', 'Intel Xeon'),
    # AMD Ryzen
    ('r3', 'AMD Ryzen 3'),
    ('r5', 'AMD Ryzen 5'),
    ('r7', 'AMD Ryzen 7'),
    ('r9', 'AMD Ryzen 9'),
    # Apple Silicon
    ('m1', 'Apple M1'),
    ('m2', 'Apple M2'),
    ('m3', 'Apple M3'),
    ('m4', 'Apple M4'),
    # Qualcomm
    ('SNAPDRAGON', 'Qualcomm Snapdragon'),
    # Unknown
    ('UNKNOWN', 'Unknown'),
]

CPU_BRAND_CHOICES = [
    ('', 'Select CPU Brand'),
    ('INTEL', 'Intel'),
    ('AMD', 'AMD'),
    ('APPLE', 'Apple'),
    ('QUALCOMM', 'Qualcomm'),
    ('UNKNOWN', 'Unknown'),
]

# Generation choices from 0 to 15 (0 = Unknown/Apple Silicon)
GENERATION_CHOICES = [
    ('0', 'Unknown / N/A (Apple)'),
    ('1', '1st Gen'),
    ('2', '2nd Gen'),
    ('3', '3rd Gen'),
    ('4', '4th Gen'),
    ('5', '5th Gen'),
    ('6', '6th Gen'),
    ('7', '7th Gen'),
    ('8', '8th Gen'),
    ('9', '9th Gen'),
    ('10', '10th Gen'),
    ('11', '11th Gen'),
    ('12', '12th Gen'),
    ('13', '13th Gen'),
    ('14', '14th Gen'),
    ('15', '15th Gen'),
]

CPU_SUFFIX_CHOICES = [
    ('', 'None/Standard'),
    ('U', 'U (Ultra-low power)'),
    ('H', 'H (High performance)'),
    ('HQ', 'HQ (High performance Quad)'),
    ('HS', 'HS (High performance Slim)'),
    ('HX', 'HX (Extreme performance)'),
    ('G', 'G (With integrated graphics)'),
    ('P', 'P (Performance)'),
    ('PRO', 'PRO (Professional)'),
    ('MAX', 'MAX (Apple Silicon Max)'),
]

GPU_TIER_CHOICES = [
    ('NONE', 'No Dedicated GPU (Integrated)'),
    # NVIDIA GeForce GTX
    ('GTX', 'GTX (Generic)'),
    ('GTX 1050', 'GTX 1050'),
    ('GTX 1060', 'GTX 1060'),
    ('GTX 1070', 'GTX 1070'),
    ('GTX 1080', 'GTX 1080'),
    ('GTX 1650', 'GTX 1650'),
    ('GTX 1660', 'GTX 1660'),
    # NVIDIA GeForce RTX 20 Series
    ('RTX 2050', 'RTX 2050'),
    ('RTX 2060', 'RTX 2060'),
    ('RTX 2070', 'RTX 2070'),
    ('RTX 2080', 'RTX 2080'),
    # NVIDIA GeForce RTX 30 Series
    ('RTX 3050', 'RTX 3050'),
    ('RTX 3060', 'RTX 3060'),
    ('RTX 3070', 'RTX 3070'),
    ('RTX 3080', 'RTX 3080'),
    # NVIDIA GeForce RTX 40 Series
    ('RTX 4050', 'RTX 4050'),
    ('RTX 4060', 'RTX 4060'),
    ('RTX 4070', 'RTX 4070'),
    ('RTX 4080', 'RTX 4080'),
    ('RTX 4090', 'RTX 4090'),
    # NVIDIA GeForce RTX 50 Series
    ('RTX 5060', 'RTX 5060'),
    ('RTX 5070', 'RTX 5070'),
    ('RTX 5080', 'RTX 5080'),
    ('RTX 5090', 'RTX 5090'),
    # NVIDIA MX Series
    ('MX', 'MX (Generic)'),
    # NVIDIA Quadro / Workstation
    ('Quadro Legacy', 'Quadro (Legacy)'),
    ('Quadro M-Series', 'Quadro M-Series'),
    ('Quadro P-Series', 'Quadro P-Series'),
    ('Quadro T-Series', 'Quadro T-Series'),
    ('RTX Workstation 2000', 'RTX Workstation 2000'),
    ('RTX Workstation 3000', 'RTX Workstation 3000'),
    ('RTX Workstation 4000', 'RTX Workstation 4000'),
    ('RTX Workstation 5000', 'RTX Workstation 5000'),
    # AMD Radeon
    ('RX 6', 'AMD RX 6000 Series'),
    ('RX 7', 'AMD RX 7000 Series'),
    # Intel Arc
    ('ARC', 'Intel Arc'),
    # Other
    ('OTHER', 'Other'),
]

GPU_SUFFIX_CHOICES = [
    ('', 'None/Standard'),
    ('TI', 'Ti'),
    ('SUPER', 'Super'),
    ('MAX-Q', 'Max-Q'),
]

CONDITION_CHOICES = [
    ('JAMAIS UTILISÉ', 'Never Used (Brand New)'),
    ('BON ÉTAT', 'Good Condition'),
    ('MOYEN', 'Average Condition'),
    ('UNKNOWN', 'Unknown'),
]

RAM_TYPE_CHOICES = [
    ('Unknown', 'Unknown'),
    ('DDR3', 'DDR3'),
    ('DDR4', 'DDR4'),
    ('LPDDR4x', 'LPDDR4x'),
    ('DDR5', 'DDR5'),
    ('LPDDR5', 'LPDDR5'),
    ('LPDDR5X', 'LPDDR5X'),
]

RESOLUTION_CLASS_CHOICES = [
    ('Unknown', 'Unknown'),
    ('HD', 'HD (1366×768)'),
    ('FHD', 'Full HD (1920×1080)'),
    ('FHD+ / WUXGA', 'FHD+ / WUXGA (1920×1200)'),
    ('QHD / 2K', 'QHD / 2K (2560×1440)'),
    ('QHD+ / 3K', 'QHD+ / 3K (2560×1600+)'),
    ('UHD / 4K+', 'UHD / 4K+ (3840×2160+)'),
]

SCREEN_SIZE_CHOICES = [
    (0, 'Unknown'),
    (13.3, '13.3"'),
    (13.6, '13.6"'),
    (14.0, '14.0"'),
    (14.2, '14.2"'),
    (15.6, '15.6"'),
    (16.0, '16.0"'),
    (16.1, '16.1"'),
    (17.3, '17.3"'),
]


class LaptopSpecsForm(forms.Form):
    # Basic Info
    brand = forms.ChoiceField(
        choices=BRAND_CHOICES,
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    series = forms.ChoiceField(
        choices=SERIES_CHOICES,
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    condition = forms.ChoiceField(
        choices=CONDITION_CHOICES,
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    
    # CPU Specifications
    cpu_brand = forms.ChoiceField(
        choices=CPU_BRAND_CHOICES,
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    cpu_family = forms.ChoiceField(
        choices=CPU_FAMILY_CHOICES,
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    cpu_generation = forms.ChoiceField(
        choices=GENERATION_CHOICES,
        required=False,
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    cpu_suffix = forms.ChoiceField(
        choices=CPU_SUFFIX_CHOICES,
        required=False,
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    
    # GPU Specifications
    gpu_tier = forms.ChoiceField(
        choices=GPU_TIER_CHOICES,
        initial='NONE',
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    gpu_suffix = forms.ChoiceField(
        choices=GPU_SUFFIX_CHOICES,
        required=False,
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    
    # Memory
    ram_size_gb = forms.IntegerField(
        min_value=2, max_value=128,
        initial=8,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 8, 16, 32'})
    )
    
    # Storage
    ssd_size_gb = forms.IntegerField(
        min_value=0, max_value=4096,
        initial=256,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 256, 512, 1024 (0 if none)'})
    )
    hdd_size_gb = forms.IntegerField(
        min_value=0, max_value=4096,
        initial=0,
        required=False,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 500, 1000 (0 if none)'})
    )
    ram_type_class = forms.ChoiceField(
        choices=RAM_TYPE_CHOICES,
        initial='Unknown',
        required=False,
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    
    # Display
    screen_size = forms.TypedChoiceField(
        choices=SCREEN_SIZE_CHOICES,
        coerce=float,
        initial=15.6,
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    resolution_class = forms.ChoiceField(
        choices=RESOLUTION_CLASS_CHOICES,
        initial='Unknown',
        required=False,
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    
    # Additional
    is_pro = forms.BooleanField(
        required=False,
        initial=False,
        label="Professional/Business Laptop",
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )
    listing_year = forms.IntegerField(
        min_value=2020, max_value=2025,
        initial=2025,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )


class PredictionFeedbackForm(forms.Form):
    """Form for collecting user feedback on predictions"""
    
    is_accurate = forms.ChoiceField(
        label="Is the prediction close/accurate?",
        choices=[
            ('', 'Select an option'),
            ('yes', 'Yes, it\'s accurate'),
            ('close', 'Close enough'),
            ('no', 'Not accurate'),
        ],
        required=False,
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    
    actual_price = forms.DecimalField(
        label="If you know the correct price, please enter it (DZD)",
        required=False,
        min_value=0,
        max_digits=12,
        decimal_places=2,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter actual price in DZD (optional)'
        })
    )
    
    feedback_text = forms.CharField(
        label="Additional feedback",
        required=False,
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 3,
            'placeholder': 'Share your thoughts about the prediction (optional)'
        })
    )

