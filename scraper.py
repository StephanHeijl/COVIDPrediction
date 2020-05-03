import scrapy
import requests
import os

MONTHS = ["januari", "februari", "maart", "april", "mei", "juni", "juli", "augustus", "september", "oktober", "november", "december"]


class RIVMSpider(scrapy.Spider):
    name = 'rivmspider'
    start_urls = ['https://www.rivm.nl/documenten?search=situatie%20COVID&document_type=All&onderwerp=&page=0%2C0%2C0']

    @staticmethod
    def convert_date(title):
        split_title = title.split("-")
        end = split_title.index("2020")
        date = split_title[end-2:end+1]
        day = int(date[0])
        month = MONTHS.index(date[1].lower())
        year = int(date[2])

        return (year, month, day)

    def parse(self, response):
        for card in response.css('.documentlist .card'):
            url_title = card.css('.file').attrib['href']
            ymd = self.convert_date(url_title)

            pdf_url = card.css('.download').attrib['href']
            pdf_path = "./pdfs/%s_%s_%s.pdf" % ymd

            if not os.path.exists(pdf_path):
                pdf_request = requests.get(pdf_url)
                with open(pdf_path, "wb+") as f:
                    f.write(pdf_request.content)

            yield {"path": pdf_path}

        for next_page in response.css('.pagination a.next'):
            yield response.follow(next_page, self.parse)