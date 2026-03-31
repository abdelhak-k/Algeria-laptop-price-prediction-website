# SEO and security middleware

class SEOHeadersMiddleware:
    """Add SEO-friendly headers to responses."""
    
    def __init__(self, get_response):
        self.get_response = get_response
        
    def __call__(self, request):
        response = self.get_response(request)
        # Add canonical URL
        response['X-Content-Type-Options'] = 'nosniff'
        
        # Add referrer policy for better privacy and tracking
        response['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        
        # Permissions policy (formerly Feature-Policy)
        response['Permissions-Policy'] = 'geolocation=(), microphone=(), camera=()'
        
        return response
