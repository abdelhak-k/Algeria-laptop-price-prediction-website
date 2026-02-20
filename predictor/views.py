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

from .forms import LaptopSpecsForm, PredictionFeedbackForm, BudgetForm
from .model_utils import predict_price, SERIES_TIER_MAP, generate_suggestions
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


# ---------- Budget Suggest ----------

def suggest(request):
    """Suggest laptops within a user-defined budget range."""
    if request.method == 'POST':
        form = BudgetForm(request.POST)
        if form.is_valid():
            min_price = form.cleaned_data['min_price']
            max_price = form.cleaned_data['max_price']
            condition = form.cleaned_data.get('condition') or None

            result = generate_suggestions(min_price, max_price, condition_filter=condition)

            return render(request, 'predictor/suggest_result.html', {
                'form': form,
                'result': result,
                'min_price': min_price,
                'max_price': max_price,
                'min_price_formatted': f"{min_price:,}",
                'max_price_formatted': f"{max_price:,}",
                'condition': condition,
            })
    else:
        form = BudgetForm()

    return render(request, 'predictor/suggest.html', {'form': form})


# ---------- Compare Laptops ----------

def _get_segment_tier(specs):
    """Segment (1=Budget, 2=Consumer, 3=Business, 4=Premium, 5=Flagship). Matches model_utils logic."""
    series = (specs.get('series') or '').strip().upper()
    brand = (specs.get('brand') or '').strip().upper()
    tier = SERIES_TIER_MAP.get(series, 2)
    if brand == 'APPLE' and tier < 4:
        tier = 4
    return tier


def _build_compare_row(feature_name, display_values, winner_from_values=None, lower_better=False, higher_better=False):
    """Build a comparison row. Only set winner when one value is strictly best (no tie)."""
    compare_vals = winner_from_values if winner_from_values is not None else display_values
    winner_idx = None
    if lower_better and compare_vals and all(v is not None for v in compare_vals):
        try:
            numeric = [float(v) if isinstance(v, (int, float)) else float(str(v).replace(',', '').replace(' ', '')) for v in compare_vals]
            min_val = min(numeric)
            min_indices = [i for i, n in enumerate(numeric) if n == min_val]
            if len(min_indices) == 1:
                winner_idx = min_indices[0]
        except (ValueError, TypeError):
            pass
    if higher_better and compare_vals and all(v is not None for v in compare_vals):
        try:
            numeric = [float(v) if isinstance(v, (int, float)) else float(str(v).replace(',', '').replace(' ', '')) for v in compare_vals]
            max_val = max(numeric)
            max_indices = [i for i, n in enumerate(numeric) if n == max_val]
            if len(max_indices) == 1:
                winner_idx = max_indices[0]
        except (ValueError, TypeError):
            pass
    return {'feature': feature_name, 'values': display_values, 'winner_idx': winner_idx}


def compare(request):
    """Compare 2 or 3 laptops: form (GET) or run predictions and show comparison (POST)."""
    num_laptops = 2
    if request.method == 'POST':
        num_laptops = int(request.POST.get('num_laptops', 2))
    else:
        try:
            num_laptops = int(request.GET.get('num', 2))
        except (ValueError, TypeError):
            num_laptops = 2
    if num_laptops not in (2, 3):
        num_laptops = 2

    prefixes = ['lap1', 'lap2', 'lap3'][:num_laptops]
    forms = [LaptopSpecsForm(prefix=p) for p in prefixes]

    if request.method == 'POST':
        for i, p in enumerate(prefixes):
            forms[i] = LaptopSpecsForm(request.POST, prefix=p)
        all_valid = all(f.is_valid() for f in forms)
        if all_valid:
            results = []
            for f in forms:
                result = predict_price(f.cleaned_data)
                results.append({
                    'specs': f.cleaned_data,
                    'prediction': result,
                })
            # Build comparison rows for the result table
            rows = []
            # Specs (display only or with winner)
            for idx, label in enumerate(['Brand', 'Series', 'Condition']):
                key = ['brand', 'series', 'condition'][idx]
                vals = [r['specs'].get(key, '—') for r in results]
                rows.append(_build_compare_row(label, vals))

            cpu_labels = [f"{r['specs'].get('cpu_brand', '')} {r['specs'].get('cpu_family', '')} Gen {r['specs'].get('cpu_generation', '0')} {r['specs'].get('cpu_suffix', '') or ''}".strip() for r in results]
            rows.append(_build_compare_row('CPU', cpu_labels))
            gpu_labels = [f"{r['specs'].get('gpu_tier', '')} {r['specs'].get('gpu_suffix', '') or ''}".strip() or 'Integrated' for r in results]
            rows.append(_build_compare_row('GPU', gpu_labels))

            ram_vals = [r['specs'].get('ram_size_gb') for r in results]
            rows.append(_build_compare_row('RAM (GB)', ram_vals, higher_better=True))
            ssd_vals = [r['specs'].get('ssd_size_gb') for r in results]
            rows.append(_build_compare_row('SSD (GB)', ssd_vals, higher_better=True))
            hdd_vals = [r['specs'].get('hdd_size_gb') for r in results]
            rows.append(_build_compare_row('HDD (GB)', hdd_vals, higher_better=True))
            screen_vals = [r['specs'].get('screen_size') for r in results]
            rows.append(_build_compare_row('Screen (")', screen_vals, higher_better=True))
            res_vals = [r['specs'].get('resolution_class', '—') for r in results]
            rows.append(_build_compare_row('Resolution', res_vals))

            # Price winner only when all laptops are in the same segment (tier)
            tiers = [_get_segment_tier(r['specs']) for r in results]
            same_segment = len(set(tiers)) == 1

            pred_prices = [r['prediction'].get('predicted_price') if r['prediction'].get('success') else None for r in results]
            pred_labels = [f"{p:,.0f} DZD" if p is not None else "—" for p in pred_prices]
            rows.append(_build_compare_row('Predicted price', pred_labels, winner_from_values=pred_prices, lower_better=same_segment))

            min_prices = [r['prediction'].get('min_price') if r['prediction'].get('success') else None for r in results]
            min_labels = [f"{p:,.0f} DZD" if p is not None else "—" for p in min_prices]
            rows.append(_build_compare_row('Price range (min)', min_labels, winner_from_values=min_prices, lower_better=same_segment))
            max_prices = [r['prediction'].get('max_price') if r['prediction'].get('success') else None for r in results]
            max_labels = [f"{p:,.0f} DZD" if p is not None else "—" for p in max_prices]
            rows.append(_build_compare_row('Price range (max)', max_labels, winner_from_values=max_prices, lower_better=same_segment))

            return render(request, 'predictor/compare_result.html', {
                'results': results,
                'num_laptops': num_laptops,
                'rows': rows,
            })
        # re-render form with errors
    return render(request, 'predictor/compare.html', {
        'forms': forms,
        'prefixes': prefixes,
        'prefixes_json': json.dumps(prefixes),
        'num_laptops': num_laptops,
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
