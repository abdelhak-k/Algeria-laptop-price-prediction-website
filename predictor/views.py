from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.contrib import messages
from django.contrib.auth import logout
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth.views import LoginView
from django.db.models import Count, Avg, Q
from django.utils import timezone
from datetime import timedelta
import json

from .forms import LaptopSpecsForm, PredictionFeedbackForm
from .model_utils import predict_price
from .models import PredictionFeedback


def _is_staff(user):
    return user.is_authenticated and user.is_staff


def home(request):
    if request.method == 'POST':
        form = LaptopSpecsForm(request.POST)
        if form.is_valid():
            form_data = form.cleaned_data
            result = predict_price(form_data)
            
            # Save the prediction + specs to the database immediately
            feedback_record = None
            if result.get('success'):
                try:
                    feedback_record = PredictionFeedback.objects.create(
                        predicted_price=result.get('predicted_price', 0),
                        min_price=result.get('min_price'),
                        max_price=result.get('max_price'),
                        
                        brand=form_data.get('brand', ''),
                        series=form_data.get('series', ''),
                        condition=form_data.get('condition', ''),
                        listing_year=form_data.get('listing_year', 2025),
                        
                        cpu_brand=form_data.get('cpu_brand', ''),
                        cpu_family=form_data.get('cpu_family', ''),
                        cpu_generation=form_data.get('cpu_generation', '0'),
                        cpu_suffix=form_data.get('cpu_suffix', ''),
                        is_pro=form_data.get('is_pro', False),
                        
                        gpu_tier=form_data.get('gpu_tier', ''),
                        gpu_suffix=form_data.get('gpu_suffix', ''),
                        
                        ram_size_gb=form_data.get('ram_size_gb', 0),
                        ram_type_class=form_data.get('ram_type_class', 'Unknown'),
                        ssd_size_gb=form_data.get('ssd_size_gb', 0),
                        hdd_size_gb=form_data.get('hdd_size_gb', 0),
                        
                        screen_size=form_data.get('screen_size', 0),
                        resolution_class=form_data.get('resolution_class', 'Unknown'),
                    )
                except Exception:
                    pass  # Don't block the user if saving fails
            
            # Create a feedback form
            feedback_form = PredictionFeedbackForm()
            
            return render(request, 'predictor/result.html', {
                'form': form,
                'result': result,
                'form_data': form_data,
                'feedback_form': feedback_form,
                'feedback_id': feedback_record.id if feedback_record else None,
            })
    else:
        form = LaptopSpecsForm()
    
    return render(request, 'predictor/home.html', {'form': form})


def submit_feedback(request):
    """Handle feedback submission — updates an existing prediction record. Returns JSON for AJAX."""
    is_ajax = (
        request.headers.get('X-Requested-With') == 'XMLHttpRequest'
        or 'application/json' in request.headers.get('Accept', '')
    )

    if request.method == 'POST':
        feedback_form = PredictionFeedbackForm(request.POST)

        if feedback_form.is_valid():
            feedback_data = feedback_form.cleaned_data
            feedback_id = request.POST.get('feedback_id')

            try:
                feedback = get_object_or_404(PredictionFeedback, id=feedback_id)

                # Update only the feedback fields
                if feedback_data.get('actual_price') is not None:
                    feedback.actual_price = feedback_data['actual_price']
                if feedback_data.get('is_accurate'):
                    feedback.is_accurate = feedback_data['is_accurate']
                if feedback_data.get('feedback_text'):
                    feedback.feedback_text = feedback_data['feedback_text']

                feedback.save()
                if is_ajax:
                    return JsonResponse({
                        'success': True,
                        'message': 'Thank you for your feedback! It will help us improve our predictions.',
                    })
                messages.success(request, 'Thank you for your feedback! It will help us improve our predictions.')
            except Exception as e:
                if is_ajax:
                    return JsonResponse({'success': False, 'error': str(e)}, status=400)
                messages.error(request, f'Error saving feedback: {str(e)}')
        else:
            if is_ajax:
                return JsonResponse({
                    'success': False,
                    'error': 'Please provide at least some feedback.',
                    'errors': feedback_form.errors,
                }, status=400)
            messages.warning(request, 'Please provide at least some feedback.')

    if is_ajax:
        return JsonResponse({'success': False, 'error': 'POST required'}, status=405)
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


# ---------- Custom Admin Dashboard ----------

@login_required(login_url='/dashboard/login/')
@user_passes_test(_is_staff, login_url='/dashboard/login/')
def dashboard(request):
    """Custom admin dashboard with charts and statistics."""
    qs = PredictionFeedback.objects.all()

    # Stats cards
    total_predictions = qs.count()
    with_feedback = qs.filter(
        Q(is_accurate__in=['yes', 'close', 'no']) |
        Q(actual_price__isnull=False) |
        ~Q(feedback_text='')
    ).distinct().count()

    accuracy_yes = qs.filter(is_accurate='yes').count()
    accuracy_close = qs.filter(is_accurate='close').count()
    accuracy_no = qs.filter(is_accurate='no').count()

    avg_predicted = qs.aggregate(avg=Avg('predicted_price'))['avg'] or 0

    # Predictions over last 14 days (for chart)
    daily = []
    for i in range(13, -1, -1):
        day = timezone.now().date() - timedelta(days=i)
        count = qs.filter(created_at__date=day).count()
        daily.append({'date': day.isoformat(), 'count': count})

    # Top brands (top 8)
    brand_counts = list(
        qs.values('brand').annotate(count=Count('id')).order_by('-count')[:8]
    )
    brands_labels = [b['brand'] for b in brand_counts]
    brands_data = [b['count'] for b in brand_counts]

    # Condition distribution
    condition_counts = list(
        qs.values('condition').annotate(count=Count('id')).order_by('-count')
    )
    condition_labels = [c['condition'] for c in condition_counts]
    condition_data = [c['count'] for c in condition_counts]

    context = {
        'total_predictions': total_predictions,
        'with_feedback': with_feedback,
        'accuracy_yes': accuracy_yes,
        'accuracy_close': accuracy_close,
        'accuracy_no': accuracy_no,
        'avg_predicted': round(float(avg_predicted), 0),
        'daily_stats_json': json.dumps(daily),
        'brands_labels_json': json.dumps(brands_labels),
        'brands_data_json': json.dumps(brands_data),
        'brands_count': len(brands_labels),
        'condition_labels_json': json.dumps(condition_labels),
        'condition_data_json': json.dumps(condition_data),
    }
    return render(request, 'predictor/dashboard.html', context)


@login_required(login_url='/dashboard/login/')
@user_passes_test(_is_staff, login_url='/dashboard/login/')
def dashboard_feedback(request):
    """List all user feedbacks with optional filter and pagination."""
    from django.core.paginator import Paginator

    qs = PredictionFeedback.objects.all().order_by('-created_at')
    filter_with_feedback = request.GET.get('with_feedback', '')
    if filter_with_feedback == '1':
        qs = qs.filter(
            Q(is_accurate__in=['yes', 'close', 'no']) |
            Q(actual_price__isnull=False) |
            ~Q(feedback_text='')
        ).distinct()

    paginator = Paginator(qs, 20)
    page_number = request.GET.get('page', 1)
    page_obj = paginator.get_page(page_number)

    context = {
        'page_obj': page_obj,
        'filter_with_feedback': filter_with_feedback,
    }
    return render(request, 'predictor/dashboard_feedback.html', context)


class DashboardLoginView(LoginView):
    """Login view for dashboard; redirects to /dashboard/ after success."""
    template_name = 'predictor/dashboard_login.html'
    redirect_authenticated_user = True

    def get_success_url(self):
        return '/dashboard/'


def dashboard_logout(request):
    """Log out and redirect to dashboard login."""
    logout(request)
    return redirect('predictor:dashboard_login')
