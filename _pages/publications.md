---
layout: page
permalink:
title: publications
description: 
years: [2015, 2020]
nav: false 
---

<div class="publications">

{% for y in page.years %}
  <h2 class="year">{{y}}</h2>
  {% bibliography -f papers -q @*[year={{y}}]* %}
{% endfor %}

</div>
