from django.db import models
from django.utils import timezone


class PredictionLog(models.Model):
    """Always store what features users entered and the prediction result"""
    
    # Prediction result
    predicted_price = models.DecimalField(max_digits=12, decimal_places=2)
    min_price = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    max_price = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    
    # Laptop specifications (stored as entered by user)
    brand = models.CharField(max_length=50)
    series = models.CharField(max_length=50)
    condition = models.CharField(max_length=50)
    listing_year = models.IntegerField()
    
    # CPU
    cpu_brand = models.CharField(max_length=50)
    cpu_family = models.CharField(max_length=50)
    cpu_generation = models.CharField(max_length=10)
    cpu_suffix = models.CharField(max_length=20, blank=True)
    is_pro = models.BooleanField(default=False)
    
    # GPU
    gpu_tier = models.CharField(max_length=50)
    gpu_suffix = models.CharField(max_length=20, blank=True)
    
    # Memory & Storage
    ram_size_gb = models.IntegerField()
    ram_type_class = models.CharField(max_length=20)
    ssd_size_gb = models.IntegerField()
    hdd_size_gb = models.IntegerField()
    
    # Display
    screen_size = models.DecimalField(max_digits=4, decimal_places=2)
    resolution_class = models.CharField(max_length=20)
    
    # Metadata
    created_at = models.DateTimeField(default=timezone.now)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Prediction Log'
        verbose_name_plural = 'Prediction Logs'
    
    def __str__(self):
        return f"Prediction - {self.brand} {self.series} - {self.predicted_price} DZD - {self.created_at.strftime('%Y-%m-%d %H:%M')}"


class PredictionFeedback(models.Model):
    """Store user feedback on price predictions"""
    
    # Link to the prediction log
    prediction = models.OneToOneField(
        PredictionLog,
        on_delete=models.CASCADE,
        related_name='feedback',
        null=True,
        blank=True,
    )
    
    # Feedback fields
    is_accurate = models.CharField(
        max_length=20,
        choices=[
            ('yes', 'Yes, it\'s accurate'),
            ('close', 'Close enough'),
            ('no', 'Not accurate'),
        ],
        null=True,
        blank=True
    )
    actual_price = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    feedback_text = models.TextField(blank=True, null=True)
    
    # Metadata
    created_at = models.DateTimeField(default=timezone.now)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Prediction Feedback'
        verbose_name_plural = 'Prediction Feedbacks'
    
    def __str__(self):
        if self.prediction:
            return f"Feedback for {self.prediction.brand} {self.prediction.series} - {self.created_at.strftime('%Y-%m-%d %H:%M')}"
        return f"Feedback #{self.pk} - {self.created_at.strftime('%Y-%m-%d %H:%M')}"
