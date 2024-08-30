#Plant-Prophet

## Inspiration
Bangladesh is a land of agriculture, where farmers are the backbone of the nation's economy. However, the daily struggles they face are often overshadowed by the unpredictability of nature, soil degradation, and lack of access to modern technology. Growing up in a country where the sight of farmers toiling in the fields is part of everyday life, we’ve witnessed the impact of failed crops, erratic weather, and infertile soil. It is these very hardships that inspired us to create Plant-Prophet, a farming application designed to empower farmers with data-driven decisions. We wanted to build something that could combine the power of technology and the wisdom of the soil to help those who nurture our food.

## What it does
Plant-Prophet is a powerful tool that provides farmers with actionable insights to improve their yield and sustainability. It has three main features:

1. **Soil Detection using ML**: Farmers can upload an image of their soil, and the app will detect the soil type using machine learning. This information is critical in determining which crops can grow best in that specific soil.
2. **Crop Recommendation using ML**: Based on the soil type and other environmental factors, the app recommends the top five crops that are most likely to thrive, helping farmers make informed decisions.
3. **Current Weather Display**: Knowing the weather is crucial for planning agricultural activities. Plant-Prophet provides real-time weather updates to help farmers stay ahead of nature’s unpredictability.

## How we built it
Building Plant-Prophet was a journey that combined multiple technologies and ideas. We started with collecting data on soil types, crop preferences, and historical weather patterns. The machine learning models for soil detection and crop recommendation were trained using reliable datasets. We used convolutional neural networks (CNNs) for soil type identification and ensemble models for crop recommendation.

For the weather feature, we integrated APIs to pull real-time weather data, ensuring that farmers have access to accurate and up-to-date information. We used Python, TensorFlow, and Scikit-learn to develop the core models, and then integrated them into a user-friendly interface that could be easily accessed by farmers.

## Challenges we ran into
With the vastly numerous soil types, it was difficult to receive enough data set to create an accurate ML model. To combat this, we reduced the number of classes to 6, namely:
1. **Alluvial**
2. **Clayey**
3. **Loamy**
4. **Peat**
5. **Red**
6. **Sandy Loam**

These soil types were chosen based on their abundance in Bangladesh as well as the difference in soil contents. 

## Accomplishments that we're proud of
We are incredibly proud of how Plant-Prophet turned out. The fact that we were able to build a fully functional application that can analyze soil, recommend crops, and provide weather updates is a testament to the hard work and dedication of our team. Seeing our models accurately predict soil types and recommend the right crops was a rewarding experience, knowing that this could potentially make a real difference in the lives of farmers. Moreover, making technology more accessible to those who need it most feels like a significant achievement.

## What we learned
Throughout the development of Plant-Prophet, we learned the importance of data—its accuracy, diversity, and relevance. We also gained a deeper understanding of the real-world challenges that farmers face and how technology can serve as a bridge to alleviate some of those hardships. Building a solution that is not only technically sound but also practical and user-friendly for people with limited resources taught us the importance of empathy in technology design.

## What's next for Plant-Prophet
The journey of Plant-Prophet doesn’t end here. We have several ideas to take this application further. We want to introduce features like irrigation advice, crop related articles, directives for planting and nurturing crops, a marketplace where farmers can directly buy and sell farming equipment and crops. We also aim to improve our machine learning models with more diverse datasets and optimize the app for offline usage, making it even more accessible to rural farmers. The future of Plant-Prophet is filled with possibilities, and we are excited to continue developing and expanding this tool to make a lasting impact.

In the end, Plant-Prophet is not just an application; it's our humble contribution to supporting those who work tirelessly to keep us fed.
