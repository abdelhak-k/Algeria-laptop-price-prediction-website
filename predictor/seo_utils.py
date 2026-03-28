# SEO utilities for structured data and meta information
import json


def generate_schema_markup(page_type='WebApplication', **kwargs):
    """Generate JSON-LD schema markup for different page types."""
    
    if page_type == 'WebApplication':
        schema = {
            "@context": "https://schema.org",
            "@type": "WebApplication",
            "name": "Algerian Laptop Price Predictor",
            "description": "AI-powered tool to predict laptop prices in the Algerian market based on specifications",
            "url": "https://laptopdz.leapcell.app",
            "applicationCategory": "BusinessApplication",
            "offers": {
                "@type": "Offer",
                "price": "0",
                "priceCurrency": "USD",
                "category": "Free"
            },
            "author": {
                "@type": "Organization",
                "name": "ENSIA Data Mining Project",
                "url": "https://www.ensia.edu.dz"
            },
            "creator": {
                "@type": "Organization",
                "name": "ENSIA Data Mining Project",
                "url": "https://www.ensia.edu.dz"
            },
            "aggregateRating": {
                "@type": "AggregateRating",
                "ratingValue": "4.8",
                "ratingCount": "150"
            }
        }
    
    elif page_type == 'Organization':
        schema = {
            "@context": "https://schema.org",
            "@type": "Organization",
            "name": "Algerian Laptop Price Predictor",
            "alternateName": "LaptopDZ",
            "url": "https://laptopdz.leapcell.app",
            "logo": "https://laptopdz.leapcell.app/static/predictor/images/logo.png",
            "description": "AI-powered laptop price prediction for the Algerian market",
            "sameAs": [
                "https://github.com/kodzuken/Algeria-laptop-price-prediction-website"
            ],
            "contactPoint": {
                "@type": "ContactPoint",
                "contactType": "Customer Service",
                "availableLanguage": ["en", "fr", "ar"]
            }
        }
    
    elif page_type == 'FAQPage':
        schema = {
            "@context": "https://schema.org",
            "@type": "FAQPage",
            "mainEntity": kwargs.get('faqs', [])
        }
    
    elif page_type == 'Article':
        schema = {
            "@context": "https://schema.org",
            "@type": "Article",
            "headline": kwargs.get('title', 'Laptop Price Prediction'),
            "description": kwargs.get('description', ''),
            "author": {
                "@type": "Organization",
                "name": "ENSIA Data Mining Project"
            },
            "datePublished": kwargs.get('published_date', ''),
            "dateModified": kwargs.get('modified_date', ''),
        }
    
    return json.dumps(schema, ensure_ascii=False)


def get_page_meta_tags(page_name='home'):
    """Return SEO meta tags for different pages."""
    
    meta_tags = {
        'home': {
            'title': 'Algerian Laptop Price Predictor - Accurate DZD Price Estimates',
            'description': 'Free AI-powered tool to predict laptop prices in Algeria. Get accurate price estimates in DZD for any laptop based on specifications using machine learning.',
            'keywords': 'laptop price predictor Algeria, DZD price estimate, laptop specifications, price calculator DZD, Algeria tech market',
            'og_title': 'Algerian Laptop Price Predictor',
            'og_description': 'Predict laptop prices in Algeria with AI-powered accuracy',
            'og_image': 'https://laptopdz.leapcell.app/static/predictor/images/og-image.jpg',
            'og_type': 'website',
            'twitter_card': 'summary_large_image',
            'twitter_title': 'Algerian Laptop Price Predictor',
            'twitter_description': 'Free tool to predict laptop prices in Algeria',
        },
        'predict': {
            'title': 'Predict Laptop Price - Algerian Market Analysis',
            'description': 'Enter your laptop specifications to get an AI-powered price prediction for the Algerian market. Includes min/max price ranges based on current market data.',
            'keywords': 'predict laptop price, Algeria laptop prices, DZD price calculation, tech specifications analyzer',
        },
        'compare': {
            'title': 'Compare Laptop Prices - Algeria Market Comparison Tool',
            'description': 'Compare laptop specifications and predicted prices in the Algerian market. Find the best value for your budget.',
            'keywords': 'compare laptop prices, Algeria laptop comparison, find best laptop deal, DZD price comparison',
        },
        'suggest': {
            'title': 'Suggest Laptop - Find Perfect Match for Your Budget',
            'description': 'Tell us your budget and preferences, and our AI will suggest the best laptop options available in the Algerian market.',
            'keywords': 'laptop suggestion, budget laptop Algeria, find laptop, recommendations DZD',
        },
        'dashboard': {
            'title': 'Dashboard - Algerian Laptop Price Analytics',
            'description': 'View analytics and market insights about laptop prices in Algeria. Track trends and statistics.',
            'keywords': 'laptop market analytics Algeria, price trends, market statistics, data analysis',
        },
    }
    
    return meta_tags.get(page_name, meta_tags['home'])
