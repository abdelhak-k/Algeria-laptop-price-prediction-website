# Sitemap configuration for SEO
from django.contrib.sitemaps import Sitemap
from django.urls import reverse


class StaticViewSitemap(Sitemap):
    """Sitemap for static pages."""
    changefreq = "weekly"
    priority = 0.9

    def items(self):
        return ['predictor:home', 'predictor:compare', 'predictor:suggest', 'predictor:dashboard']

    def location(self, item):
        return reverse(item)


class DynamicPagesSitemap(Sitemap):
    """Sitemap for dynamic pages."""
    changefreq = "daily"
    priority = 0.7

    def items(self):
        # These represent page patterns, not actual items
        return [
            {'name': 'home', 'priority': 0.9},
            {'name': 'predict', 'priority': 0.8},
            {'name': 'compare', 'priority': 0.8},
            {'name': 'suggest', 'priority': 0.7},
            {'name': 'dashboard', 'priority': 0.7},
        ]

    def location(self, item):
        return reverse('predictor:home')


sitemaps = {
    'static': StaticViewSitemap,
    'dynamic': DynamicPagesSitemap,
}
