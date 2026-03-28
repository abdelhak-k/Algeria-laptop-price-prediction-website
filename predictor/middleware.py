# SEO and security middleware
from django.middleware.base import MiddlewareBase


class SEOHeadersMiddleware(MiddlewareBase):
    """Add SEO-friendly headers to responses."""
    
    def process_response(self, request, response):
        # Add canonical URL
        response['X-Content-Type-Options'] = 'nosniff'
        
        # Add referrer policy for better privacy and tracking
        response['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        
        # Permissions policy (formerly Feature-Policy)
        response['Permissions-Policy'] = 'geolocation=(), microphone=(), camera=()'
        
        return response
