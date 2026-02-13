from django.contrib import admin
from .models import PredictionFeedback


@admin.register(PredictionFeedback)
class PredictionFeedbackAdmin(admin.ModelAdmin):
    list_display = ['id', 'brand', 'series', 'predicted_price', 'actual_price', 'is_accurate', 'created_at']
    list_filter = ['is_accurate', 'brand', 'condition', 'created_at']
    search_fields = ['brand', 'series', 'feedback_text']
    readonly_fields = ['created_at']
    date_hierarchy = 'created_at'
    
    fieldsets = (
        ('Prediction Info', {
            'fields': ('predicted_price', 'actual_price', 'is_accurate')
        }),
        ('Feedback', {
            'fields': ('feedback_text',)
        }),
        ('Laptop Specifications', {
            'fields': (
                'brand', 'series', 'condition', 'listing_year',
                'cpu_brand', 'cpu_family', 'cpu_generation', 'cpu_suffix', 'is_pro',
                'gpu_tier', 'gpu_suffix',
                'ram_size_gb', 'ram_type_class',
                'ssd_size_gb', 'hdd_size_gb',
                'screen_size', 'resolution_class'
            )
        }),
        ('Metadata', {
            'fields': ('created_at',)
        }),
    )

