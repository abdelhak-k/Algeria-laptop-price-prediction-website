from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib import messages
from .forms import LaptopSpecsForm, PredictionFeedbackForm
from .model_utils import predict_price
from .models import PredictionLog, PredictionFeedback


def home(request):
    if request.method == 'POST':
        form = LaptopSpecsForm(request.POST)
        if form.is_valid():
            form_data = form.cleaned_data
            result = predict_price(form_data)
            
            # Always save the prediction log
            prediction_log = None
            if result.get('success'):
                try:
                    prediction_log = PredictionLog.objects.create(
                        predicted_price=result['predicted_price'],
                        min_price=result.get('min_price'),
                        max_price=result.get('max_price'),
                        brand=form_data.get('brand', ''),
                        series=form_data.get('series', ''),
                        condition=form_data.get('condition', ''),
                        listing_year=form_data.get('listing_year', 2025),
                        cpu_brand=form_data.get('cpu_brand', ''),
                        cpu_family=form_data.get('cpu_family', ''),
                        cpu_generation=form_data.get('cpu_generation', '0') or '0',
                        cpu_suffix=form_data.get('cpu_suffix', '') or '',
                        is_pro=form_data.get('is_pro', False),
                        gpu_tier=form_data.get('gpu_tier', 'NONE') or 'NONE',
                        gpu_suffix=form_data.get('gpu_suffix', '') or '',
                        ram_size_gb=form_data.get('ram_size_gb', 8),
                        ram_type_class=form_data.get('ram_type_class', 'Unknown') or 'Unknown',
                        ssd_size_gb=form_data.get('ssd_size_gb', 0) or 0,
                        hdd_size_gb=form_data.get('hdd_size_gb', 0) or 0,
                        screen_size=form_data.get('screen_size', 15.6),
                        resolution_class=form_data.get('resolution_class', 'Unknown') or 'Unknown',
                    )
                except Exception:
                    pass  # Don't break the prediction if logging fails
            
            # Create a feedback form
            feedback_form = PredictionFeedbackForm()
            
            return render(request, 'predictor/result.html', {
                'form': form,
                'result': result,
                'form_data': form_data,
                'feedback_form': feedback_form,
                'prediction_log_id': prediction_log.pk if prediction_log else None,
            })
    else:
        form = LaptopSpecsForm()
    
    return render(request, 'predictor/home.html', {'form': form})


def submit_feedback(request):
    """Handle feedback submission"""
    if request.method == 'POST':
        feedback_form = PredictionFeedbackForm(request.POST)
        
        if feedback_form.is_valid():
            feedback_data = feedback_form.cleaned_data
            
            # Check that at least some feedback was provided
            has_feedback = (
                feedback_data.get('is_accurate') or
                feedback_data.get('actual_price') is not None or
                feedback_data.get('feedback_text')
            )
            
            if not has_feedback:
                messages.warning(request, 'Please provide at least some feedback.')
                return redirect('predictor:home')
            
            try:
                # Get the prediction log ID from the hidden field
                prediction_log_id = request.POST.get('prediction_log_id')
                prediction_log = None
                if prediction_log_id:
                    try:
                        prediction_log = PredictionLog.objects.get(pk=prediction_log_id)
                    except PredictionLog.DoesNotExist:
                        pass
                
                feedback = PredictionFeedback.objects.create(
                    prediction=prediction_log,
                    is_accurate=feedback_data.get('is_accurate') or None,
                    actual_price=feedback_data.get('actual_price'),
                    feedback_text=feedback_data.get('feedback_text') or '',
                )
                
                messages.success(request, 'Thank you for your feedback! It will help us improve our predictions.')
                
            except Exception as e:
                messages.error(request, f'Error saving feedback: {str(e)}')
        else:
            messages.warning(request, 'Please provide valid feedback.')
    
    return redirect('predictor:home')


def predict_api(request):
    if request.method == 'POST':
        form = LaptopSpecsForm(request.POST)
        if form.is_valid():
            form_data = form.cleaned_data
            result = predict_price(form_data)
            
            # Also log API predictions
            if result.get('success'):
                try:
                    PredictionLog.objects.create(
                        predicted_price=result['predicted_price'],
                        min_price=result.get('min_price'),
                        max_price=result.get('max_price'),
                        brand=form_data.get('brand', ''),
                        series=form_data.get('series', ''),
                        condition=form_data.get('condition', ''),
                        listing_year=form_data.get('listing_year', 2025),
                        cpu_brand=form_data.get('cpu_brand', ''),
                        cpu_family=form_data.get('cpu_family', ''),
                        cpu_generation=form_data.get('cpu_generation', '0') or '0',
                        cpu_suffix=form_data.get('cpu_suffix', '') or '',
                        is_pro=form_data.get('is_pro', False),
                        gpu_tier=form_data.get('gpu_tier', 'NONE') or 'NONE',
                        gpu_suffix=form_data.get('gpu_suffix', '') or '',
                        ram_size_gb=form_data.get('ram_size_gb', 8),
                        ram_type_class=form_data.get('ram_type_class', 'Unknown') or 'Unknown',
                        ssd_size_gb=form_data.get('ssd_size_gb', 0) or 0,
                        hdd_size_gb=form_data.get('hdd_size_gb', 0) or 0,
                        screen_size=form_data.get('screen_size', 15.6),
                        resolution_class=form_data.get('resolution_class', 'Unknown') or 'Unknown',
                    )
                except Exception:
                    pass
            
            return JsonResponse(result)
        else:
            return JsonResponse({
                'success': False,
                'error': 'Invalid form data',
                'errors': form.errors
            })
    
    return JsonResponse({
        'success': False,
        'error': 'POST method required'
    })

