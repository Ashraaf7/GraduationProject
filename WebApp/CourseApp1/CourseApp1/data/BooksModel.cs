using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Web;

namespace CourseApp1.data
{
    [MetadataType(typeof(BookMetadata))]
    public partial class Book
    {

    }
    public class BookMetadata
    {
        public int BookId { get; set; }
        [Display(Name = "Book Category")]
        public int BookCategory { get; set; }
        [Display(Name = "Book link")]
        public string BookLink { get; set; }
        [Display(Name = "Book Title")]
        public string BookTitle { get; set; }
        public string Authors { get; set; }
    }
}