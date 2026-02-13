from django.contrib import admin
from .models import PredictionLog, PredictionFeedback


class PredictionFeedbackInline(admin.StackedInline):
    model = PredictionFeedback
    extra = 0
    readonly_fields = ['created_at']
    can_delete = False


@admin.register(PredictionLog)
class PredictionLogAdmin(admin.ModelAdmin):
    list_display = ['id', 'brand', 'series', 'predicted_price', 'cpu_family', 'gpu_tier', 'ram_size_gb', 'has_feedback', 'created_at']
    list_filter = ['brand', 'condition', 'cpu_brand', 'gpu_tier', 'created_at']
    search_fields = ['brand', 'series', 'cpu_family', 'gpu_tier']
    readonly_fields = ['created_at']
    date_hierarchy = 'created_at'
    inlines = [PredictionFeedbackInline]
    
    fieldsets = (
        ('Prediction Result', {
            'fields': ('predicted_price', 'min_price', 'max_price')
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
    
    @admin.display(boolean=True, description='Has Feedback')
    def has_feedback(self, obj):
        return hasattr(obj, 'feedback') and obj.feedback is not None


@admin.register(PredictionFeedback)
class PredictionFeedbackAdmin(admin.ModelAdmin):
    list_display = ['id', 'get_brand', 'get_series', 'get_predicted_price', 'actual_price', 'is_accurate', 'created_at']
    list_filter = ['is_accurate', 'created_at']
    search_fields = ['feedback_text', 'prediction__brand', 'prediction__series']
    readonly_fields = ['created_at', 'prediction']
    date_hierarchy = 'created_at'
    
    fieldsets = (
        ('Linked Prediction', {
            'fields': ('prediction',)
        }),
        ('Feedback', {
            'fields': ('is_accurate', 'actual_price', 'feedback_text')
        }),
        ('Metadata', {
            'fields': ('created_at',)
        }),
    )
    
    @admin.display(description='Brand')
    def get_brand(self, obj):
        return obj.prediction.brand if obj.prediction else '-'
    
    @admin.display(description='Series')
    def get_series(self, obj):
        return obj.prediction.series if obj.prediction else '-'
    
    @admin.display(description='Predicted Price')
    def get_predicted_price(self, obj):
        return obj.prediction.predicted_price if obj.prediction else '-'
