from django.shortcuts import render
from django.http import JsonResponse
from .forms import LaptopSpecsForm
from .model_utils import predict_price


def home(request):
    if request.method == 'POST':
        form = LaptopSpecsForm(request.POST)
        if form.is_valid():
            form_data = form.cleaned_data
            result = predict_price(form_data)
            
            return render(request, 'predictor/result.html', {
                'form': form,
                'result': result,
                'form_data': form_data
            })
    else:
        form = LaptopSpecsForm()
    
    return render(request, 'predictor/home.html', {'form': form})


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
