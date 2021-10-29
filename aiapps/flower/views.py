from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from .forms import PhotoForm
from .models import Photo

def index(request):
    template = loader.get_template('flower/index.html')
    context = {'form': PhotoForm()}
    return HttpResponse(template.render(context, request))


def predict(request):
    if not request.method == 'POST':
        return
        redirect('flower:index')

    form = PhotoForm(request.POST, request.FILES)
    if not form.is_valid():
        raise ValuueError('Formが不正です')
    
    photo = Photo(image=form.cleaned_data['image'])
    predicted, ratio = photo.predict()

    template = loader.get_template('flower/result.html')

    context = {
        'photo_name': photo.image.name,
        'photo_data': photo.image_src(),
        'predicted': predicted,
        'ratio': ratio,
    }
    return HttpResponse(template.render(context, request))

