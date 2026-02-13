from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib import messages
from .forms import LaptopSpecsForm, PredictionFeedbackForm
from .model_utils import predict_price
from .models import PredictionFeedback


def home(request):
    if request.method == 'POST':
        form = LaptopSpecsForm(request.POST)
        if form.is_valid():
            form_data = form.cleaned_data
            result = predict_price(form_data)
            
            # Create a feedback form
            feedback_form = PredictionFeedbackForm()
            
            return render(request, 'predictor/result.html', {
                'form': form,
                'result': result,
                'form_data': form_data,
                'feedback_form': feedback_form,
            })
    else:
        form = LaptopSpecsForm()
    
    return render(request, 'predictor/home.html', {'form': form})


def submit_feedback(request):
    """Handle feedback submission"""
    if request.method == 'POST':
        feedback_form = PredictionFeedbackForm(request.POST)
        
        if feedback_form.is_valid():
            # Get feedback data
            feedback_data = feedback_form.cleaned_data
            
            # Get the laptop specs and prediction from POST data
            # These are passed as hidden fields from the result page
            try:
                feedback = PredictionFeedback(
                    predicted_price=request.POST.get('predicted_price'),
                    actual_price=feedback_data.get('actual_price'),
                    is_accurate=feedback_data.get('is_accurate'),
                    feedback_text=feedback_data.get('feedback_text'),
                    
                    # Laptop specifications
                    brand=request.POST.get('brand'),
                    series=request.POST.get('series'),
                    condition=request.POST.get('condition'),
                    listing_year=request.POST.get('listing_year'),
                    
                    cpu_brand=request.POST.get('cpu_brand'),
                    cpu_family=request.POST.get('cpu_family'),
                    cpu_generation=request.POST.get('cpu_generation'),
                    cpu_suffix=request.POST.get('cpu_suffix', ''),
                    is_pro=request.POST.get('is_pro') == 'True',
                    
                    gpu_tier=request.POST.get('gpu_tier'),
                    gpu_suffix=request.POST.get('gpu_suffix', ''),
                    
                    ram_size_gb=request.POST.get('ram_size_gb'),
                    ram_type_class=request.POST.get('ram_type_class'),
                    ssd_size_gb=request.POST.get('ssd_size_gb'),
                    hdd_size_gb=request.POST.get('hdd_size_gb'),
                    
                    screen_size=request.POST.get('screen_size'),
                    resolution_class=request.POST.get('resolution_class'),
                )
                
                feedback.save()
                messages.success(request, 'Thank you for your feedback! It will help us improve our predictions.')
                
            except Exception as e:
                messages.error(request, f'Error saving feedback: {str(e)}')
        else:
            messages.warning(request, 'Please provide at least some feedback.')
    
    return redirect('predictor:home')


def predict_api(request):
    if request.method == 'POST':
        form = LaptopSpecsForm(request.POST)
        if form.is_valid():
            result = predict_price(form.cleaned_data)
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
