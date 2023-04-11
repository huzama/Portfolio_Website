---
layout: page
title: experience
permalink: /experience/
display_categories: [Work, Volunteer]
nav: true
nav_order: 3
---

<div class="experience">
    <h2 class="category">{{ page.display_categories[0] }}</h2>
    {% for work in site.data.experience.work %}
      {% include experience/work.html work=work %}
    {% endfor %}

<!--    <h2 class="category">{{ page.display_categories[1] }}</h2>
    {% for volunteer in site.data.experience.volunteer %}
      {% include experience/volunteer.html volunteer=volunteer %}
    {% endfor %} -->
</div>